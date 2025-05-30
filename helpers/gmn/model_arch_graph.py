# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import math
from copy import deepcopy
import torch
import torch.nn as nn
import torch_geometric

from helpers.gmn.constants import NODE_TYPES, EDGE_TYPES, CONV_LAYERS, NORM_LAYERS, RESIDUAL_LAYERS
from helpers.gmn.utils import make_node_feat, make_edge_attr, conv_to_graph, linear_to_graph, norm_to_graph, ffn_to_graph, basic_block_to_graph, self_attention_to_graph, equiv_set_linear_to_graph, triplanar_to_graph
from helpers.gmn.layers import Flatten, PositionwiseFeedForward, BasicBlock, SelfAttention, EquivSetLinear, TriplanarGrid

# model <--> arch <--> graph

def sequential_to_arch(model):
    # input can be a nn.Sequential
    # or ordered list of modules
    arch = []
    weight_bias_modules = CONV_LAYERS + [nn.Linear] + NORM_LAYERS
    for module in model:
        layer = [type(module)]
        if type(module) in weight_bias_modules:
            layer.append(module.weight)
            layer.append(module.bias)
        elif type(module) == BasicBlock:
            layer.extend([
                module.conv1.weight,
                module.bn1.weight,
                module.bn1.bias,
                module.conv2.weight,
                module.bn2.weight,
                module.bn2.bias])
            if len(module.shortcut) > 0:
                layer.extend([
                    module.shortcut[0].weight,
                    module.shortcut[1].weight,
                    module.shortcut[1].bias])
        elif type(module) == PositionwiseFeedForward:
            layer.append(module.lin1.weight)
            layer.append(module.lin1.bias)
            layer.append(module.lin2.weight)
            layer.append(module.lin2.bias)
        elif type(module) == SelfAttention:
            layer.append(module.attn.in_proj_weight)
            layer.append(module.attn.in_proj_bias)
            layer.append(module.attn.out_proj.weight)
            layer.append(module.attn.out_proj.bias)
        elif type(module) == EquivSetLinear:
            layer.append(module.lin1.weight)
            layer.append(module.lin1.bias)
            layer.append(module.lin2.weight)
        elif type(module) == TriplanarGrid:
            layer.append(module.tgrid)
        else:
            if len(list(module.parameters())) != 0:
                raise ValueError(f'{type(module)} has parameters but is not yet supported')
            continue
        arch.append(layer)
    return arch

def arch_to_graph(arch, self_loops=False):
    
    curr_idx = 0
    x = []
    edge_index = []
    edge_attr = []
    layer_num = 0
    
    # initialize input nodes
    layer = arch[0]
    if layer[0] in CONV_LAYERS:
        in_neuron_idx = torch.arange(layer[1].shape[1])
    elif layer[0] in (nn.Linear, PositionwiseFeedForward):
        in_neuron_idx = torch.arange(layer[1].shape[1])
    elif layer[0] == BasicBlock:
        in_neuron_idx = torch.arange(layer[1].shape[1])
    elif layer[0] == EquivSetLinear:
        in_neuron_idx = torch.arange(layer[1].shape[1])
    elif layer[0] == TriplanarGrid:
        triplanar_resolution = layer[1].shape[2]
        in_neuron_idx = torch.arange(3*triplanar_resolution**2)
    else:
        raise ValueError('Invalid first layer')
    
    for i, layer in enumerate(arch):
        out_neuron = (i==len(arch)-1)
        if layer[0] in CONV_LAYERS:
            ret = conv_to_graph(layer[1], layer[2], layer_num, in_neuron_idx, out_neuron, curr_idx, self_loops)
            layer_num += 1
        elif layer[0] == nn.Linear:
            ret = linear_to_graph(layer[1], layer[2], layer_num, in_neuron_idx, out_neuron, curr_idx, self_loops)
            layer_num += 1
        elif layer[0] in NORM_LAYERS:
            if layer[0] in (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d):
                norm_type = 'bn'
            elif layer[0] == nn.LayerNorm:
                norm_type = 'ln'
            elif layer[0] == nn.GroupNorm:
                norm_type = 'gn'
            elif layer[0] in (nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d):
                norm_type = 'in'
            else:
                raise ValueError('Invalid norm type')
            ret = norm_to_graph(layer[1], layer[2], layer_num, in_neuron_idx, out_neuron, curr_idx, self_loops, norm_type=norm_type)
        elif layer[0] == BasicBlock:
            ret = basic_block_to_graph(layer[1:], layer_num, in_neuron_idx, out_neuron, curr_idx, self_loops)
            layer_num += 2
        elif layer[0] == PositionwiseFeedForward:
            ret = ffn_to_graph(layer[1], layer[2], layer[3], layer[4], layer_num, in_neuron_idx, out_neuron, curr_idx, self_loops)
            layer_num += 2
        elif layer[0] == SelfAttention:
            ret = self_attention_to_graph(layer[1], layer[2], layer[3], layer[4], layer_num, in_neuron_idx, out_neuron=out_neuron, curr_idx=curr_idx, self_loops=self_loops)
            layer_num += 2
        elif layer[0] == EquivSetLinear:
            ret = equiv_set_linear_to_graph(layer[1], layer[2], layer[3], layer_num, in_neuron_idx, out_neuron=out_neuron, curr_idx=curr_idx, self_loops=self_loops)
            layer_num += 1
        elif layer[0] == TriplanarGrid:
            ret = triplanar_to_graph(layer[1], layer_num, out_neuron=out_neuron, curr_idx=curr_idx)
            layer_num += 1
        else:
            raise ValueError('Invalid layer type')
        in_neuron_idx = ret['out_neuron_idx']
            
        edge_index.append(ret['edge_index'])
        edge_attr.append(ret['edge_attr'])
        if ret['added_x'] is not None:
            feat = ret['added_x']
            x.append(feat)
            curr_idx += feat.shape[0]

    x = torch.cat(x, dim=0)
    edge_index = torch.cat(edge_index, dim=1)
    edge_attr = torch.cat(edge_attr, dim=0)
    return x, edge_index, edge_attr

def graph_to_arch(arch, weights):
    # arch is the original arch
    arch_new = []
    curr_idx = 0
    for l, layer in enumerate(arch):
        lst = [layer[0]]
        if layer[0] != SelfAttention:
            for tensor in layer[1:]:
                if tensor is not None:
                    weight_size = math.prod(tensor.shape)
                    reshaped = weights[curr_idx:curr_idx+weight_size].reshape(tensor.shape)
                    lst.append(reshaped)
                    curr_idx += weight_size
                else:
                    lst.append(None)
        else:
            # handle in_proj stuff differently, because pytorch stores it all as a big matrix
            in_proj_weight_shape = layer[1].shape
            dim = in_proj_weight_shape[1]
            in_proj_weight = []
            in_proj_bias = []
            for _ in range(3):
                # get q, k, and v
                weight_size = dim*dim
                reshaped = weights[curr_idx:curr_idx+weight_size].reshape(dim, dim)
                in_proj_weight.append(reshaped)
                curr_idx += weight_size
                
                bias_size = dim
                reshaped = weights[curr_idx:curr_idx+bias_size].reshape(dim)
                in_proj_bias.append(reshaped)
                curr_idx += bias_size 
                
            # concatenate q, k, v weights and biases
            lst.append(torch.cat(in_proj_weight, 0))
            lst.append(torch.cat(in_proj_bias, 0))
            
            # out_proj handled normally
            for tensor in layer[3:]:
                if tensor is not None:
                    weight_size = math.prod(tensor.shape)
                    reshaped = weights[curr_idx:curr_idx+weight_size].reshape(tensor.shape)
                    lst.append(reshaped)
                    curr_idx += weight_size
                else:
                    lst.append(None)
        
        # handle residual connections, and other edges that don't correspond to weights
        if layer[0] == PositionwiseFeedForward:
            residual_size = layer[1].shape[1]
            curr_idx += residual_size
        elif layer[0] == BasicBlock:
            residual_size = layer[1].shape[0]
            curr_idx += residual_size
        elif layer[0] == SelfAttention:
            residual_size = layer[1].shape[1]
            curr_idx += residual_size
            
        arch_new.append(lst)
    return arch_new

def arch_to_sequential(arch, model):
    # model is a model of the correct architecture
    arch_idx = 0
    for child in model.children():
        if len(list(child.parameters())) > 0:
            layer = arch[arch_idx]
            sd = child.state_dict()
            layer_idx = 1
            for i, k in enumerate(sd):
                if 'running_mean' in k or 'running_var' in k or 'num_batches_tracked' in k:
                    continue
                param = nn.Parameter(layer[layer_idx])
                sd[k] = param
                layer_idx += 1
            child.load_state_dict(sd)
            arch_idx += 1
    return model