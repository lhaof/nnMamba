import torch
import torch.nn.init as init
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0., step_size=0.1):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(dim, dim))
        with torch.no_grad():
            init.kaiming_uniform_(self.weight)
        self.step_size = step_size
        self.lambd = 0.1

    def forward(self, x):
        # compute D^T * D * x
        x1 = F.linear(x, self.weight, bias=None)
        grad_1 = F.linear(x1, self.weight.t(), bias=None)
        # compute D^T * x
        grad_2 = F.linear(x, self.weight.t(), bias=None)
        # compute negative gradient update: step_size * (D^T * x - D^T * D * x)
        grad_update = self.step_size * (grad_2 - grad_1) - self.step_size * self.lambd

        output = F.relu(x + grad_update)
        return output

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.qkv = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        w = rearrange(self.qkv(x), 'b n (h d) -> b h n d', h = self.heads)

        dots = torch.matmul(w, w.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, w)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

import torch.nn.functional as F
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, dropout = 0., ista=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.heads = heads
        self.depth = depth
        self.dim = dim
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, dim, dropout = dropout, step_size=ista))
            ]))

    def forward(self, x):
        depth = 0
        for attn, ff in self.layers:
            grad_x = attn(x) + x

            x = ff(grad_x)
        return x

class CRATE(nn.Module):
    def __init__(self, *, image_size, patch_size, frames, frame_patch_size, num_classes, dim, depth, heads, pool = 'cls', channels = 1, dim_head = 64, dropout = 0., emb_dropout = 0., ista=0.1):
        super().__init__()

        # self.features = nn.Sequential(OrderedDict([
        #     ('conv0', nn.Conv3d(1, 64,
        #                         kernel_size=7, stride=2, padding=3, bias=False)),
        #     ('norm0', nn.BatchNorm3d(64)),
        #     ('relu0', nn.ReLU(inplace=True)),
        #     ('pool0', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
        # ]))

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        # self.pos_embed = pos_embed

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert frames % frame_patch_size == 0, 'Frames must be divisible by frame patch size'

        num_patches = (image_height // patch_height) * (image_width // patch_width) * (frames // frame_patch_size)
        patch_dim = channels * patch_height * patch_width * frame_patch_size
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        # if self.pos_embed == 'perceptron':
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (f pf) (w p2) -> b (h f w) (p1 pf p2 c)', p1 = patch_height, p2 = patch_width, pf = frame_patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        # elif self.pos_embed == "conv":
        #     self.to_patch_embedding = Conv[Conv.CONV, 3](
        #         in_channels=channels, out_channels=dim, kernel_size=patch_size, stride=patch_size
        #     )


        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # self.trunc_normal_(self.pos_embedding, mean=0.0, std=0.02, a=-2.0, b=2.0)
        # self.apply(self._init_weights)

        self.transformer = Transformer(dim, depth, heads, dim_head, dropout, ista=ista)

        self.pool = pool
        self.to_latent = nn.Identity()

        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(dim),
        #     nn.Linear(dim, num_classes)
        # )

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1000),
            nn.Linear(1000, num_classes)
        )

        self.sigmoid = nn.Sigmoid()

    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         self.trunc_normal_(m.weight, mean=0.0, std=0.02, a=-2.0, b=2.0)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)

    # def trunc_normal_(self, tensor, mean, std, a, b):
    #     # From PyTorch official master until it's in a few official releases - RW
    #     # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    #     def norm_cdf(x):
    #         return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    #     with torch.no_grad():
    #         l = norm_cdf((a - mean) / std)
    #         u = norm_cdf((b - mean) / std)
    #         tensor.uniform_(2 * l - 1, 2 * u - 1)
    #         tensor.erfinv_()
    #         tensor.mul_(std * math.sqrt(2.0))
    #         tensor.add_(mean)
    #         tensor.clamp_(min=a, max=b)
    #         return tensor


    def forward(self, img):
        # img = self.features(img)
        # print('img', img.shape)

        x = self.to_patch_embedding(img)
        # if self.pos_embed == "conv":
        #     x = x.flatten(2).transpose(-1, -2)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)
        feature_pre = x
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        feature_last = x
        x = self.mlp_head(x)
        x = self.sigmoid(x)
        return x


def CRATE_tiny(num_classes = 1000):
    return CRATE(image_size=224,
                    patch_size=16,
                    num_classes=num_classes,
                    dim=384,
                    depth=12,
                    heads=6,
                    dropout=0.0,
                    emb_dropout=0.0,
                    dim_head=384//6)

def CRATE_small(num_classes = 1000):
    return CRATE(image_size=224,
                    patch_size=16,
                    num_classes=num_classes,
                    dim=576,
                    depth=12,
                    heads=12,
                    dropout=0.0,
                    emb_dropout=0.0,
                    dim_head=576//12)

def CRATE_base(num_classes = 1000):
    return CRATE(image_size=224,
                patch_size=16,
                num_classes=num_classes,
                dim=768,
                depth=12,
                heads=12,
                dropout=0.0,
                emb_dropout=0.0,
                dim_head=768//12)

def CRATE_large(num_classes = 1000):
    return CRATE(image_size=224,
                patch_size=16,
                num_classes=num_classes,
                dim=1024,
                depth=24,
                heads=16,
                dropout=0.0,
                emb_dropout=0.0,
                dim_head=1024//16)

def CRATE_base_3D(num_classes = 1):
    return CRATE(image_size=96,
                patch_size=16,
                frames=96,
                frame_patch_size=16,
                num_classes=num_classes,
                dim=768,
                depth=12,
                heads=12,
                dropout=0.0,
                emb_dropout=0.0,
                dim_head=768//12)

def CRATE_small_3D(num_classes = 1):
    return CRATE(image_size=104,
                patch_size=8,
                frames=128,
                frame_patch_size=8,
                num_classes=num_classes,
                dim=576,
                depth=12,
                heads=12,
                dropout=0.0,
                emb_dropout=0.0,
                dim_head=576//12)

# def CRATE_small_3D(num_classes = 1):
#     return CRATE(image_size=32,
#                 patch_size=4,
#                 frames=28,
#                 frame_patch_size=4,
#                 num_classes=num_classes,
#                 dim=576,
#                 depth=12,
#                 heads=12,
#                 dropout=0.0,
#                 emb_dropout=0.0,
#                 dim_head=576//12)

def CRATE_tiny_3D(num_classes = 1):
    return CRATE(image_size=102,
                patch_size=8,
                frames=128,
                frame_patch_size=8,
                num_classes=num_classes,
                dim=384,
                depth=12,
                heads=6,
                dropout=0.0,
                emb_dropout=0.0,
                dim_head=384//6,
                )


if __name__ == '__main__':
    # test model check the output size
    model = CRATE_small_3D().cuda()
    x = torch.randn(1,1,112,126,112).cuda()
    output = model(x)
    print(output.shape)