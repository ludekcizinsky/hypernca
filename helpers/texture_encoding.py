import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from torchvision.models.vgg import VGG16_Weights
from torchvision.models.vision_transformer import VisionTransformer as AbstractViT
from torch import Tensor
from torchvision.models.vision_transformer import vit_b_16
from transformers import CLIPProcessor, CLIPModel
import torch.nn as nn
import torch

class CLIP(nn.Module):
    def __init__(self, 
            model_name:str="openai/clip-vit-base-patch32", 
            proj_dim:int=None) -> None:
        super().__init__()

        self.clip = CLIPModel.from_pretrained(model_name).to('cuda')

        self.processor = CLIPProcessor.from_pretrained(model_name,use_fast=True,do_rescale=False)

        self.proj_dim = proj_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if proj_dim is not None:
            self.projector = nn.Linear(self.clip.visual_projection.out_features, proj_dim)

        self.clip.eval()  # freeze model

        # Print number of model parameters
        num_params = sum(p.numel() for p in self.parameters())
        print(f"Number of parameters: {num_params / 1e6:.2f}M")
        
    def forward(self, x):
        # x: (B, 3, 128, 128) → resize + normalize using processor
        inputs = self.processor(images=x, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_embeds = self.clip.get_image_features(**inputs)  # (B, D)
        if self.proj_dim is not None:
            image_embeds = self.projector(image_embeds) # (B,, D)
        return image_embeds.unsqueeze(1)  # (B, 1, D)


class VisionTransformerFromScratch(AbstractViT):
    def __init__(self,patch_size:int=16,hidden_dim:int=256,num_layers:int=6,num_heads:int=8,mlp_dim:int=512,image_size:int=128) -> None:
        super(VisionTransformerFromScratch,self).__init__(
            image_size=image_size,
            patch_size=patch_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_dim=mlp_dim
            )
        self.heads = None

    def forward(self, x:Tensor):
        """
        Args:
            x: (B, 3, H, W), H=W=128
            Assuming Normalized input in [0,1] range

        Returns:
            x: (B, 1, hidden_dim)
        """
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)

        cls_token = x[:, 0]
        return cls_token.unsqueeze(1)  # (B, 1, hidden_dim)

class PretrainedVisionTransformer(nn.Module):
    def __init__(self, hidden_size:int=256,num_hidden_layers:int=0) -> None:
        """
        Using a pretrained ViT model (ViT-B/16) from torchvision.

        Args:
            hidden_size: The size of the hidden dimension.
            num_hidden_layers: The number of hidden layers used to project fro 768 to hidden_size. If 0, no projection is used.
        """
        super(PretrainedVisionTransformer, self).__init__()
        self.vit = vit_b_16(weights='DEFAULT')
        self.vit.heads = None

        self.num_hidden_layers = num_hidden_layers

        # If input is not of size 224x224, we need to resize the input
        self.input_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
        ])

        if num_hidden_layers > 0:
            self.proj = nn.ModuleList([nn.Sequential(
                nn.Linear(768 if _==0 else hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size)
            ) for _ in range(num_hidden_layers)])
            self.proj = nn.Sequential(*self.proj)


    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W), H=W=128

        Returns:
            x: (B, 1, hidden_dim)
        """
        if x.shape[2] != 224 or x.shape[3] != 224:
            # Resize the input tensor to 224x224
            x = self.input_transform(x)

        x = self.vit._process_input(x)

        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.vit.encoder(x)

        # Extract the class token
        cls_token = x[:, 0]

        if self.num_hidden_layers > 0:
            cls_token = self.proj(cls_token)
        return cls_token.unsqueeze(1)  # (B, 1, hidden_dim)


class VisionTransformer(nn.Module):
    def __init__(self,
                pretrained:bool=False,
                trainable:bool=False,
                num_hidden_layers:int=0,
                patch_size:int=16,
                hidden_dim:int=256,
                num_layers:int=6,
                num_heads:int=8,
                mlp_dim:int=512,
                image_size:int=128) -> None:
        super(VisionTransformer, self).__init__()
        if pretrained:
            self.model = PretrainedVisionTransformer(hidden_size=hidden_dim, num_hidden_layers=num_hidden_layers)
            if not trainable:
                for name,param in self.model.named_parameters():
                    if 'vit' in name:
                        param.requires_grad = False
        else:
            self.model = VisionTransformerFromScratch(
                patch_size=patch_size,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                image_size=image_size
            )
            # Always trainable for scratch models

        # Print number of model parameters
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {num_params / 1e6:.2f}M")

        # Normalize the input
        self.input_transform = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.output_dim = 768 if pretrained and num_hidden_layers == 0 else hidden_dim

    def forward(self, x:Tensor):
        """
        Args:
            x: (B, 3, H, W), H=W=128
            Assuming Normalized input in [0,1] range

        Returns:
            x: (B, 1, hidden_dim)
        """
        #x = self.input_transform(x)
        x = self.model(x)
        return x

class GramEncoder(nn.Module):
    def __init__(self, hidden_size:int=512, normalize=False, num_tokens=1, **kwargs) -> None:
        super(GramEncoder, self).__init__()
        self.vgg16_features = models.vgg16(weights=VGG16_Weights.DEFAULT).features

        for param in self.vgg16_features.parameters():
            param.requires_grad = False

        self.normalize = normalize
        self.num_tokens = num_tokens

        self.style_layers = [1, 6, 11, 18, 25]  # 1, 6, 11, 18, 25
        self.n_grams = len(self.style_layers)

        if normalize:
            self.normalizing_stds = [16.0, 3.79, 1.48, 0.98, 1.84] * self.n_grams
        else:
            self.normalizing_stds = [1.0] * self.n_grams

        self.gram_embedding = nn.ParameterList(
            [
                nn.Parameter(
                    torch.randn(c, hidden_size // self.n_grams)
                    * 0.01
                    * self.normalizing_stds[i],
                    requires_grad=True,
                )
                for i, c in enumerate([64, 128, 256, 512, 512])
            ]
        )

    def qi_extractor(self, feature, param):
        q = param.T @ feature
        q = torch.mean(q ** 2, dim=-1)
        return q

    def forward(self, x):
        with torch.no_grad():

            mean = torch.tensor([0.485, 0.456, 0.406], device=x.device)[:, None, None]
            std = torch.tensor([0.229, 0.224, 0.225], device=x.device)[:, None, None]
            x = (x - mean) / std
            b, c, h, w = x.shape
            features = []
            for i, layer in enumerate(self.vgg16_features[: max(self.style_layers) + 1]):
                x = layer(x)
                if i in self.style_layers:
                    b, c, h, w = x.size()
                    feature = x.view(b, c, w * h)
                    features.append(feature)

        qis = []
        for i, feature in enumerate(features):
            qis.append(self.qi_extractor(feature, self.gram_embedding[i]))

        if self.num_tokens > 1:
            return torch.stack(qis, dim=1)
        else:
            return torch.cat(qis, dim=1).unsqueeze(1)
        




if __name__ == "__main__":
    #Example usage
    model = GramEncoder(hidden_size=512, normalize=True, num_tokens=1)
    x = torch.randn(2, 3, 128, 128) 
    output = model(x)
    print(output.shape)  # Should be (2, 1, 510)


    model = VisionTransformer(pretrained=True, trainable=False, num_hidden_layers=0)
    x = torch.randn(2, 3, 128, 128)
    output = model(x)
    print(output.shape)  # Should be (2, 1, 768)
