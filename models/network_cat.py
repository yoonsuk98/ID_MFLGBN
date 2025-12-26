import torch
import torch.nn as nn

from timm.models.layers import DropPath
from einops import rearrange

def img2windows(img, H_sp, W_sp):
    """
    Input: Image (B, C, H, W)
    Output: Window Partition (B', N, C)
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp* W_sp, C)
    return img_perm


def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    Input: Window Partition (B', N, C)
    Output: Image (B, H, W, C)
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
    def flops(self,x):
        
        shape = x.shape
        # B, N, C  if x is 3D;  B, C  if x is 2D
        if x.ndim == 3:
            B, N, C = shape
        elif x.ndim == 2:
            B, N, C = shape[0], 1, shape[1]
        else:
            raise ValueError(f"Unsupported input dim: {x.ndim}")

        hidden = self.fc1.out_features   # = hidden_features
        out = self.fc2.out_features      # = out_features

        # fc1: B * N * (in_features * hidden_features)
        flops_fc1 = B * N * C * hidden
        # fc2: B * N * (hidden_features * out_features)
        flops_fc2 = B * N * hidden * out

        return flops_fc1 + flops_fc2


class DynamicPosBias(nn.Module):
    # The implementation builds on Crossformer code https://github.com/cheerss/CrossFormer/blob/main/models/crossformer.py
    """ Dynamic Relative Position Bias.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        residual (bool):  If True, use residual strage to connect conv.
    """
    def __init__(self, dim, num_heads, residual):
        super().__init__()
        self.residual = residual
        self.num_heads = num_heads
        self.pos_dim = dim // 4
        self.pos_proj = nn.Linear(2, self.pos_dim)
        self.pos1 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos2 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim)
        )
        self.pos3 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.num_heads)
        )
    def forward(self, biases):
        if self.residual:
            pos = self.pos_proj(biases) # 2Gh-1 * 2Gw-1, heads
            pos = pos + self.pos1(pos)
            pos = pos + self.pos2(pos)
            pos = self.pos3(pos)
        else:
            pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos
    
    def flops(self, biases):
        # biases: 2Gh-1 * 2Gw-1, heads
        flops = 0
        flops += biases.shape[0] * biases.shape[1] * self.pos_dim


class Attention_axial(nn.Module):
    """ Axial Rectangle-Window (axial-Rwin) self-attention with dynamic relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        resolution (int): Input resolution.
        idx (int): The identix of V-Rwin and H-Rwin, -1 is Full Attention, 0 is V-Rwin, 1 is H-Rwin.
        split_size (int): Height or Width of the regular rectangle window, the other is H or W (axial-Rwin).
        dim_out (int | None): The dimension of the attention output, if None dim_out is dim. Default: None
        num_heads (int): Number of attention heads. Default: 6
        attn_drop (float): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float): Dropout ratio of output. Default: 0.0
        qk_scale (float | None): Override default qk scale of head_dim ** -0.5 if set
        position_bias (bool): The dynamic relative position bias. Default: True
    """
    def __init__(self, dim, resolution, idx, split_size=7, dim_out=None, num_heads=6, attn_drop=0., proj_drop=0., qk_scale=None, position_bias=True):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.resolution = resolution
        self.split_size = split_size
        self.num_heads = num_heads
        self.idx = idx
        self.position_bias = position_bias

        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5
        if idx == -1:
            H_sp, W_sp = self.resolution, self.resolution
        elif idx == 0:
            H_sp, W_sp = self.resolution, self.split_size
        elif idx == 1:
            W_sp, H_sp = self.resolution, self.split_size
        else:
            print ("ERROR MODE", idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp
        self.tmp_H = H_sp
        self.tmp_W = W_sp

        if self.position_bias:
            self.pos = DynamicPosBias(self.dim // 4, self.num_heads, residual=False)

        self.attn_drop = nn.Dropout(attn_drop)

    def im2win(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(-2,-1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.H_sp, self.W_sp)
        x = x.reshape(-1, self.H_sp* self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def forward(self, qkv, H, W, mask=None):
        """
        Input: qkv: (B, 3*L, C), H, W, mask: (B, N, N), N is the window size
        Output: x (B, H, W, C)
        """
        q,k,v = qkv[0], qkv[1], qkv[2]
        # the side of axial rectangle window changes with input
        if self.resolution != H or self.resolution != W:
            if self.idx == -1:
                H_sp, W_sp = H, W
            elif self.idx == 0:
                H_sp, W_sp = H, self.split_size
            elif self.idx == 1:
                W_sp, H_sp = W, self.split_size
            else:
                print ("ERROR MODE", self.idx)
                exit(0)
            self.H_sp = H_sp
            self.W_sp = W_sp
        else:
            self.H_sp = self.tmp_H
            self.W_sp = self.tmp_W

        B, L, C = q.shape
        assert L == H * W, "flatten img_tokens has wrong size"

        # partition the q,k,v, image to window
        q = self.im2win(q, H, W)
        k = self.im2win(k, H, W)
        v = self.im2win(v, H, W)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1).contiguous())  # B head N C @ B head C N --> B head N N

        # calculate drpe
        if self.position_bias:
            # generate mother-set
            position_bias_h = torch.arange(1 - self.H_sp, self.H_sp, device=attn.device)
            position_bias_w = torch.arange(1 - self.W_sp, self.W_sp, device=attn.device)
            biases = torch.stack(torch.meshgrid([position_bias_h, position_bias_w], indexing='ij')) # for pytorch >= 1.10
            # biases = torch.stack(torch.meshgrid([position_bias_h, position_bias_w])) # for pytorch < 1.10
            biases = biases.flatten(1).transpose(0, 1).contiguous().float()

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.H_sp, device=attn.device)
            coords_w = torch.arange(self.W_sp, device=attn.device)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij')) # for pytorch >= 1.10
            # coords = torch.stack(torch.meshgrid([coords_h, coords_w])) # for pytorch < 1.10
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.H_sp - 1
            relative_coords[:, :, 1] += self.W_sp - 1
            relative_coords[:, :, 0] *= 2 * self.W_sp - 1
            relative_position_index = relative_coords.sum(-1)

            pos = self.pos(biases)
            assert relative_position_index.max() < pos.shape[0], "relative_position_index가 pos의 크기를 초과!"
            
            
            # select position bias
            relative_position_bias = pos[relative_position_index.view(-1)].view(
                self.H_sp * self.W_sp, self.H_sp * self.W_sp, -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            attn = attn + relative_position_bias.unsqueeze(0)

        N = attn.shape[3]

        # use mask for shift window
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)

        x = (attn @ v)

        x = x.transpose(1, 2).contiguous().reshape(-1, self.H_sp* self.W_sp, C)  # B head N N @ B head N C

        # merge the window, window to image
        x = windows2img(x, self.H_sp, self.W_sp, H, W)  # B H' W' C

        return x
    
    def flops(self,x):
        B, C, H, W = x.shape
        H_sp, W_sp = self.H_sp, self.W_sp
        N = H_sp * W_sp
        num_windows = B * (H * W) // N
        heads = self.num_heads
        C_head = C // heads

        # 1) Q·K^T
        flops_qk = num_windows * heads * N * N * C_head
        # 2) softmax (~ N^2 additions + exponentials)
        flops_sm = num_windows * heads * N * N
        # 3) add position bias
        flops_bias = num_windows * heads * N * N
        # 4) Attn·V
        flops_av = flops_qk

        return flops_qk + flops_sm + flops_bias + flops_av
        
        

class CATB_axial(nn.Module):
    """ Axial Cross Aggregation Transformer Block.
    Args:
        dim (int): Number of input channels.
        reso (int): Input resolution.
        num_heads (int): Number of attention heads.
        split_size (int): Height or Width of the axial rectangle window, the other is H or W (axial-Rwin).
        shift_size (int): Shift size for axial-Rwin.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None): Override default qk scale of head_dim ** -0.5 if set.
        drop (float): Dropout rate. Default: 0.0
        attn_drop (float): Attention dropout rate. Default: 0.0
        drop_path (float): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module): Activation layer. Default: nn.GELU
        norm_layer (nn.Module): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, reso, num_heads,
                 split_size=7, shift_size=0, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.patches_resolution = reso
        self.split_size = split_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm1 = norm_layer(dim)

        assert 0 <= self.shift_size < self.split_size, "shift_size must in 0-split_size"

        self.branch_num = 2

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        self.attns = nn.ModuleList([
                    Attention_axial(
                        dim//2, resolution=self.patches_resolution, idx = i,
                        split_size=split_size, num_heads=num_heads//2, dim_out=dim//2,
                        qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, position_bias=True)
                    for i in range(self.branch_num)])

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)

        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,groups=dim) # DW Conv

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.patches_resolution, self.patches_resolution)
            self.register_buffer("attn_mask_0", attn_mask[0])
            self.register_buffer("attn_mask_1", attn_mask[1])
        else:
            attn_mask = None
            self.register_buffer("attn_mask_0", None)
            self.register_buffer("attn_mask_1", None)

    def calculate_mask(self, H, W):
        # The implementation builds on Swin Transformer code https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
        # calculate attention mask for Rwin
        img_mask_0 = torch.zeros((1, H, self.split_size, 1))
        img_mask_1 = torch.zeros((1, self.split_size, W, 1))
        slices = (slice(-self.split_size, -self.shift_size),
                  slice(-self.shift_size, None))
        cnt = 0
        for s in slices:
            img_mask_0[:, :, s, :] = cnt
            img_mask_1[:, s, :, :] = cnt
            cnt += 1

        # calculate mask for V-Shift
        img_mask_0 = img_mask_0.view(1, H // H, H, self.split_size // self.split_size, self.split_size, 1)
        img_mask_0 = img_mask_0.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, self.split_size, 1)
        mask_windows_0 = img_mask_0.view(-1, H * self.split_size)
        attn_mask_0 = mask_windows_0.unsqueeze(1) - mask_windows_0.unsqueeze(2)
        attn_mask_0 = attn_mask_0.masked_fill(attn_mask_0 != 0, float(-100.0)).masked_fill(attn_mask_0 == 0, float(0.0))
        num_v = W // self.split_size
        attn_mask_0_la = torch.zeros((num_v,H * self.split_size,H * self.split_size))
        attn_mask_0_la[-1] = attn_mask_0

        # calculate mask for H-Shift
        img_mask_1 = img_mask_1.view(1, self.split_size // self.split_size, self.split_size, W // W, W, 1)
        img_mask_1 = img_mask_1.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.split_size, W, 1)
        mask_windows_1 = img_mask_1.view(-1, self.split_size * W)
        attn_mask_1 = mask_windows_1.unsqueeze(1) - mask_windows_1.unsqueeze(2)
        attn_mask_1 = attn_mask_1.masked_fill(attn_mask_1 != 0, float(-100.0)).masked_fill(attn_mask_1 == 0, float(0.0))
        num_h = H // self.split_size
        attn_mask_1_la = torch.zeros((num_h,W * self.split_size,W * self.split_size))
        attn_mask_1_la[-1] = attn_mask_1

        return attn_mask_0_la, attn_mask_1_la

    def forward(self, x, x_size):
        """
        Input: x: (B, H*W, C), x_size: (H, W)
        Output: x: (B, H*W, C)
        """
        H , W = x_size
        B, L, C = x.shape
        assert L == H * W, "flatten img_tokens has wrong size"
        img = self.norm1(x)
        qkv = self.qkv(img).reshape(B, -1, 3, C).permute(2, 0, 1, 3) # 3, B, HW, C
        # v without partition
        v = qkv[2].transpose(-2,-1).contiguous().view(B, C, H, W)

        if self.shift_size > 0:
            qkv = qkv.view(3, B, H, W, C)
            # V-Shift
            qkv_0 = torch.roll(qkv[:,:,:,:,:C//2], shifts=-self.shift_size, dims=3)
            qkv_0 = qkv_0.view(3, B, L, C//2)
            # H-Shift
            qkv_1 = torch.roll(qkv[:,:,:,:,C//2:], shifts=-self.shift_size, dims=2)
            qkv_1 = qkv_1.view(3, B, L, C//2)

            if self.patches_resolution != H or self.patches_resolution != W:
                mask_tmp = self.calculate_mask(H, W)
                # V-Rwin
                x1_shift = self.attns[0](qkv_0, H, W, mask=mask_tmp[0].to(x.device))
                # H-Rwin
                x2_shift = self.attns[1](qkv_1, H, W, mask=mask_tmp[1].to(x.device))

            else:
                # V-Rwin
                x1_shift = self.attns[0](qkv_0, H, W, mask=self.attn_mask_0)
                # H-Rwin
                x2_shift = self.attns[1](qkv_1, H, W, mask=self.attn_mask_1)

            x1 = torch.roll(x1_shift, shifts=self.shift_size, dims=2)
            x2 = torch.roll(x2_shift, shifts=self.shift_size, dims=1)
            x1 = x1.view(B, L, C//2).contiguous()
            x2 = x2.view(B, L, C//2).contiguous()
            # Concat
            attened_x = torch.cat([x1,x2], dim=2)
        else:
            # V-Rwin
            x1 = self.attns[0](qkv[:,:,:,:C//2], H, W).view(B, L, C//2).contiguous()
            # H-Rwin
            x2 = self.attns[1](qkv[:,:,:,C//2:], H, W).view(B, L, C//2).contiguous()
            # Concat
            attened_x = torch.cat([x1,x2], dim=2)

        # Locality Complementary Module
        lcm = self.get_v(v)
        lcm = lcm.permute(0, 2, 3, 1).contiguous().view(B, L, C)

        attened_x = attened_x + lcm

        attened_x = self.proj(attened_x)
        attened_x = self.proj_drop(attened_x)

        x = x + self.drop_path(attened_x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x
    
    def flops(self,x):
        flops = 0
        B, C, H, W = x.shape
        M = self.mlp_ratio  # MLP 확장 비율
        # 1) qkv
        flops += 3 * C * C * H * W
        # 2) axial attention
        flops += 0.5 * C * H * W * (H + W)
        # 3) DW-Conv
        flops += 9 * C * H * W
        # 4) proj
        flops += C * C * H * W
        # 5) MLP
        flops += 2 * M * C * C * H * W
        
        return flops    


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.embed_dim = embed_dim

    def forward(self, x):
        x = self.proj(x)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()

        return x
    
    def flops(self,x):
        _, _, H, W = x.shape
        flops = self.embed_dim * H * W * 3 * 3 * 3
        return flops


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x, H, W):
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W).contiguous()
        x = self.body(x)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        return x
    
    def flops(self, x, H, W):
        B, L, C_in = x.shape
        assert L == H * W, "입력 토큰 개수와 H*W가 맞지 않습니다."
        # conv2d parameters
        C_out = C_in // 2
        kernel_size = 3

        # FLOPs for Conv2d (multiplications only, add는 동일 수로 가정)
        # B * C_out * C_in * K * K * H * W
        flops_conv = B * C_out * C_in * kernel_size * kernel_size * H * W

        # PixelUnshuffle / rearrange 는 0 FLOPs
        return flops_conv


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x, H, W):
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W).contiguous()
        x = self.body(x)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        return x
    
    def flops(self, x, H, W):
        """
        Estimate FLOPs for the convolution inside Upsample.
        Args:
            x: Tensor of shape (B, H*W, C_in)
            H, W: spatial height and width before upsampling
        Returns:
            int: total multiply-add FLOPs
        """
        B, L, C_in = x.shape
        assert L == H * W, f"Expected L==H*W but got L={L}, H*W={H*W}"
        C_out = C_in * 2
        kernel_size = 3

        # Conv2d FLOPs: B * C_out * C_in * (K*K) * H * W
        flops_conv = B * C_out * C_in * kernel_size * kernel_size * H * W

        # PixelShuffle / rearrange: 0 FLOPs
        return flops_conv
    



# The implementation builds on Restormer code https://github.com/swz30/Restormer/blob/main/basicsr/models/archs/restormer_arch.py
class CAT_Unet(nn.Module):
    def __init__(self,
                img_size=64,
                in_chans=3,
                dim=180,
                depth=[2,2,2,2],
                split_size_0 = [0,0,0,0],
                split_size_1 = [0,0,0,0],
                num_heads=[2,2,2,2],
                mlp_ratio=2.,
                num_refinement_blocks=4,
                bias=False,
                dual_pixel_task=False,
                **kwargs
    ):

        super(CAT_Unet, self).__init__()

        out_channels = in_chans
        qkv_bias = True
        qk_scale = None
        drop_rate = 0.
        attn_drop_rate = 0.
        drop_path_rate = 0.
        act_layer = nn.GELU
        norm_layer = nn.LayerNorm
        img_range = 1.

        self.patch_embed = OverlapPatchEmbed(in_chans, dim)
        self.encoder_level1 = nn.ModuleList([CATB_axial(dim=dim,
                                                           num_heads=num_heads[0],
                                                           reso=img_size,
                                                           mlp_ratio=mlp_ratio,
                                                           qkv_bias=qkv_bias,
                                                           qk_scale=qk_scale,
                                                           split_size=split_size_0[0],
                                                           drop=drop_rate,
                                                           attn_drop=attn_drop_rate,
                                                           drop_path=drop_path_rate,
                                                           act_layer=act_layer,
                                                           norm_layer=norm_layer,
                                                           shift_size=0 if (i % 2 == 0) else split_size_0[0]//2,
                                                           )
                                            for i in range(depth[0])])

        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.ModuleList([CATB_axial(dim=int(dim*2**1),
                                                           num_heads=num_heads[1],
                                                           reso=img_size,
                                                           mlp_ratio=mlp_ratio,
                                                           qkv_bias=qkv_bias,
                                                           qk_scale=qk_scale,
                                                           split_size=split_size_0[1],
                                                           drop=drop_rate,
                                                           attn_drop=attn_drop_rate,
                                                           drop_path=drop_path_rate,
                                                           act_layer=act_layer,
                                                           norm_layer=norm_layer,
                                                           shift_size=0 if (i % 2 == 0) else split_size_0[1]//2,
                                                           )
                                            for i in range(depth[1])])

        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = nn.ModuleList([CATB_axial(dim=int(dim*2**2),
                                                           num_heads=num_heads[2],
                                                           reso=img_size,
                                                           mlp_ratio=mlp_ratio,
                                                           qkv_bias=qkv_bias,
                                                           qk_scale=qk_scale,
                                                           split_size=split_size_0[2],
                                                           drop=drop_rate,
                                                           attn_drop=attn_drop_rate,
                                                           drop_path=drop_path_rate,
                                                           act_layer=act_layer,
                                                           norm_layer=norm_layer,
                                                           shift_size=0 if (i % 2 == 0) else split_size_0[2]//2,
                                                           )
                                            for i in range(depth[2])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = nn.ModuleList([CATB_axial(dim=int(dim*2**3),
                                                           num_heads=num_heads[3],
                                                           reso=img_size,
                                                           mlp_ratio=mlp_ratio,
                                                           qkv_bias=qkv_bias,
                                                           qk_scale=qk_scale,
                                                           split_size=split_size_0[3],
                                                           drop=drop_rate,
                                                           attn_drop=attn_drop_rate,
                                                           drop_path=drop_path_rate,
                                                           act_layer=act_layer,
                                                           norm_layer=norm_layer,
                                                           shift_size=0 if (i % 2 == 0) else split_size_0[3]//2,
                                                           )
                                            for i in range(depth[3])])

        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.ModuleList([CATB_axial(dim=int(dim*2**2),
                                                           num_heads=num_heads[2],
                                                           reso=img_size,
                                                           mlp_ratio=mlp_ratio,
                                                           qkv_bias=qkv_bias,
                                                           qk_scale=qk_scale,
                                                           split_size=split_size_0[2],
                                                           drop=drop_rate,
                                                           attn_drop=attn_drop_rate,
                                                           drop_path=drop_path_rate,
                                                           act_layer=act_layer,
                                                           norm_layer=norm_layer,
                                                           shift_size=0 if (i % 2 == 0) else split_size_0[2]//2,
                                                           )
                                            for i in range(depth[2])])

        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.ModuleList([CATB_axial(dim=int(dim*2**1),
                                                           num_heads=num_heads[1],
                                                           reso=img_size,
                                                           mlp_ratio=mlp_ratio,
                                                           qkv_bias=qkv_bias,
                                                           qk_scale=qk_scale,
                                                           split_size=split_size_0[1],
                                                           drop=drop_rate,
                                                           attn_drop=attn_drop_rate,
                                                           drop_path=drop_path_rate,
                                                           act_layer=act_layer,
                                                           norm_layer=norm_layer,
                                                           shift_size=0 if (i % 2 == 0) else split_size_0[1]//2,
                                                           )
                                            for i in range(depth[1])])

        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.ModuleList([CATB_axial(dim=int(dim*2**1),
                                                           num_heads=num_heads[0],
                                                           reso=img_size,
                                                           mlp_ratio=mlp_ratio,
                                                           qkv_bias=qkv_bias,
                                                           qk_scale=qk_scale,
                                                           split_size=split_size_0[0],
                                                           drop=drop_rate,
                                                           attn_drop=attn_drop_rate,
                                                           drop_path=drop_path_rate,
                                                           act_layer=act_layer,
                                                           norm_layer=norm_layer,
                                                           shift_size=0 if (i % 2 == 0) else split_size_0[0]//2,
                                                           )
                                            for i in range(depth[0])])

        self.refinement = nn.ModuleList([CATB_axial(dim=int(dim*2**1),
                                                           num_heads=num_heads[0],
                                                           reso=img_size,
                                                           mlp_ratio=mlp_ratio,
                                                           qkv_bias=qkv_bias,
                                                           qk_scale=qk_scale,
                                                           split_size=split_size_0[0],
                                                           drop=drop_rate,
                                                           attn_drop=attn_drop_rate,
                                                           drop_path=drop_path_rate,
                                                           act_layer=act_layer,
                                                           norm_layer=norm_layer,
                                                           shift_size=0 if (i % 2 == 0) else split_size_0[0]//2,
                                                           )
                                            for i in range(num_refinement_blocks)])

        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = False
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim*2**1), kernel_size=1, bias=bias)
        ###########################

        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):
        _, _, H, W = inp_img.shape
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = inp_enc_level1
    
        for layer in self.encoder_level1:
            out_enc_level1 = layer(out_enc_level1, [H, W])
        # out_enc_level1 = self.encoder_level1(inp_enc_level1)
        
        
        inp_enc_level2 = self.down1_2(out_enc_level1, H, W)
        out_enc_level2 = inp_enc_level2
        for layer in self.encoder_level2:
            out_enc_level2 = layer(out_enc_level2, [H//2, W//2])
        # out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2, H//2, W//2)
        out_enc_level3 = inp_enc_level3
        for layer in self.encoder_level3:
            out_enc_level3 = layer(out_enc_level3, [H//4, W//4])
        # out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3, H//4, W//4)
        latent = inp_enc_level4
        for layer in self.latent:
            latent = layer(latent, [H//8, W//8])
        # latent = self.latent(inp_enc_level4)

        inp_dec_level3 = self.up4_3(latent, H//8, W//8)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 2)
        inp_dec_level3 = rearrange(inp_dec_level3, "b (h w) c -> b c h w", h=H//4, w=W//4).contiguous()
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        inp_dec_level3 = rearrange(inp_dec_level3, "b c h w -> b (h w) c")
        # inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = inp_dec_level3
        for layer in self.decoder_level3:
            out_dec_level3 = layer(out_dec_level3, [H//4, W//4])
        # out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3, H//4, W//4)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 2)
        inp_dec_level2 = rearrange(inp_dec_level2, "b (h w) c -> b c h w", h=H//2, w=W//2).contiguous()
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        inp_dec_level2 = rearrange(inp_dec_level2, "b c h w -> b (h w) c")
        # inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = inp_dec_level2
        for layer in self.decoder_level2:
            out_dec_level2 = layer(out_dec_level2, [H//2, W//2])
        # out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2, H//2, W//2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 2)
        out_dec_level1 = inp_dec_level1
        for layer in self.decoder_level1:
            out_dec_level1 = layer(out_dec_level1, [H, W])
        # out_dec_level1 = self.decoder_level1(inp_dec_level1)

        for layer in self.refinement:
            out_dec_level1 = layer(out_dec_level1, [H, W])
        # out_dec_level1 = self.refinement(out_dec_level1)

        out_dec_level1 = rearrange(out_dec_level1, "b (h w) c -> b c h w", h=H, w=W)

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img


        return out_dec_level1
    
    def flops(self, inp_img):
        """
        Estimate total FLOPs for one forward pass of CAT_Unet.
        Args:
            inp_img: Tensor of shape (B, in_chans, H, W)
        Returns:
            total_flops: int
        """
        B, Cin, H, W = inp_img.shape
        total = 0

        # ---------------------------------------------------------------------
        # helper for Conv2d FLOPs
        def conv_flops(conv: nn.Conv2d, h: int, w: int, b: int):
            Cout, Cin, kH, kW = conv.weight.shape
            return b * Cin * Cout * kH * kW * h * w

        # 1) Patch embedding (assumes OverlapPatchEmbed has .proj: Conv2d)
        proj = self.patch_embed.proj
        total += conv_flops(proj, H, W, B)

        # track shapes & dims after each stage
        H1, W1 = H, W
        C1 = proj.out_channels

        # 2) Encoder Level1
        for blk in self.encoder_level1:
            # blk.flops expects x as (B, C, H, W)
            total += blk.flops(torch.zeros(B, C1, H1, W1))
        # save output1 for skip‐connection
        # (실제로 forward에서는 tokens 형태지만, flops 계산엔 필요 없음)

        # 3) Down1→2
        total += self.down1_2.flops(torch.zeros(B, C1, H1, W1), H1, W1)
        H2, W2 = H1 // 2, W1 // 2
        C2 = self.encoder_level2[0].dim  # = dim * 2**1

        # 4) Encoder Level2
        for blk in self.encoder_level2:
            total += blk.flops(torch.zeros(B, C2, H2, W2))

        # 5) Down2→3
        total += self.down2_3.flops(torch.zeros(B, C2, H2, W2), H2, W2)
        H3, W3 = H2 // 2, W2 // 2
        C3 = self.encoder_level3[0].dim  # = dim * 2**2

        # 6) Encoder Level3
        for blk in self.encoder_level3:
            total += blk.flops(torch.zeros(B, C3, H3, W3))

        # 7) Down3→4 (latent)
        total += self.down3_4.flops(torch.zeros(B, C3, H3, W3), H3, W3)
        H4, W4 = H3 // 2, W3 // 2
        C4 = self.latent[0].dim  # = dim * 2**3

        # 8) Latent blocks
        for blk in self.latent:
            total += blk.flops(torch.zeros(B, C4, H4, W4))

        # ---------------------------------------------------------------------
        # Decoder
        # 9) Up4→3
        total += self.up4_3.flops(torch.zeros(B, C4, H4, W4), H4, W4)
        # 10) channel reduction conv @ Level3
        total += conv_flops(self.reduce_chan_level3, H3, W3, B)
        C3d = self.decoder_level3[0].dim  # should == C3

        # 11) Decoder Level3
        for blk in self.decoder_level3:
            total += blk.flops(torch.zeros(B, C3d, H3, W3))

        # 12) Up3→2
        total += self.up3_2.flops(torch.zeros(B, C3d, H3, W3), H3, W3)
        # 13) channel reduction conv @ Level2
        total += conv_flops(self.reduce_chan_level2, H2, W2, B)
        C2d = self.decoder_level2[0].dim  # should == C2

        # 14) Decoder Level2
        for blk in self.decoder_level2:
            total += blk.flops(torch.zeros(B, C2d, H2, W2))

        # 15) Up2→1
        total += self.up2_1.flops(torch.zeros(B, C2d, H2, W2), H2, W2)
        C1d = self.decoder_level1[0].dim  # should == C1

        # 16) Decoder Level1
        for blk in self.decoder_level1:
            total += blk.flops(torch.zeros(B, C1d, H1, W1))

        # 17) Refinement blocks
        for blk in self.refinement:
            total += blk.flops(torch.zeros(B, C1d, H1, W1))

        # ---------------------------------------------------------------------
        # Output convolution
        total += conv_flops(self.output, H1, W1, B)

        # Dual-pixel task skip‐conv (optional)
        if self.dual_pixel_task:
            total += conv_flops(self.skip_conv, H1, W1, B)

        return total
        



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


import time
from thop import profile
from thop import clever_format

if __name__ == '__main__':

    upscale = 4
    window_size = 8
    height = 128
    width = 128
    # height = (1024 // upscale // window_size + 1) * window_size
    # width = (720 // upscale // window_size + 1) * window_size


    start_time = time.time()
    # 모델 및 입력 데이터 설정
    model = CAT_Unet(
        img_size= 128,
        in_chans= 3,
        depth= [2, 2, 4, 2, 4, 2, 2, 2],
        split_size_0= [2, 2, 2, 2, 2, 2, 2, 2],
        split_size_1= [0, 0, 0, 0, 0, 0, 0, 0],
        dim= 16,
        num_heads= [2, 2, 4, 8, 4, 4, 2, 2],
        mlp_ratio= 2,
        num_refinement_blocks= 2,
        bias= False,
        dual_pixel_task= False,
        ).cuda()
    x = torch.ones((1, 3, height, width)).cuda()
    y = model(x)
    print("--- %s seconds ---" % (time.time() - start_time))
    print(f"input shape : {x.shape}")
    print(f"model shape : {y.shape}")

    ##########################################################################################################
    # GPU 시간 측정 시작
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start_event.record()
    # 모델 추론
    yy = model(x)
    # GPU 시간 측정 종료
    end_event.record()
    torch.cuda.synchronize()
    # 소요 시간 계산
    time_taken = start_event.elapsed_time(end_event)
    print(f"Elapsed time on GPU: {time_taken} ms -> {time_taken * 1e-3} s")  # 밀리초(ms) -> 초(s) 변환
    ##########################################################################################################

    print(f"count parameter : {count_parameters(model)}")

    print(f"flops2 : {model.flops()/1e9}")
    flops, params = profile(model, inputs=(x,), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")

    print(f"FLOPs: {flops}")
    print(f"Parameters: {params}")