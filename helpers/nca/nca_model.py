import torch

def depthwise_conv(x, filters):
    """filters: [filter_n, h, w]"""
    b, ch, h, w = x.shape
    y = x.reshape(b * ch, 1, h, w)
    y = torch.nn.functional.pad(y, [1, 1, 1, 1], 'circular')
    y = torch.nn.functional.conv2d(y, filters[:, None])
    return y.reshape(b, -1, h, w)

class NCA(torch.nn.Module):
    def __init__(self, chn=12, fc_dim=96,device:str='cuda:0') -> None:
        super().__init__()
        self.chn = chn
        self.w1 = torch.nn.Conv2d(chn * 4, fc_dim, 1,bias=True)
        self.w2 = torch.nn.Conv2d(fc_dim, chn, 1, bias=False)
        torch.nn.init.xavier_normal_(self.w1.weight, gain=0.2)
        torch.nn.init.zeros_(self.w2.weight)
        self.device = device


        ident = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
        #lap = torch.tensor([[0.5, 0.0, 0.5], [2.0, -6.0, 2.0], [0.5, 0.0, 0.5]])#
        lap = torch.tensor([[1.0, 2.0, 1.0], [2.0, -12, 2.0], [1.0, 2.0, 1.0]])
        self.filters = torch.stack([ident, sobel_x, sobel_x.T, lap]).to(self.device)

    def perception(self, x):
        return depthwise_conv(x, self.filters)

    def forward(self, x, update_rate=0.5, return_hidden=False):
        y = self.perception(x)
        hidden = torch.relu(self.w1(y))
        y = self.w2(hidden)
        b, c, h, w = y.shape
        update_mask = (torch.rand(b, 1, h, w, device=self.device) + update_rate).floor()
        
        if return_hidden:
            return x + y * update_mask, hidden
        else:
            return x + y * update_mask

    def seed(self, n, size=128):
        return torch.zeros(n, self.chn, size, size)

    def to_rgb(self, x):
        return x[..., :3, :, :] + 0.5
    
    