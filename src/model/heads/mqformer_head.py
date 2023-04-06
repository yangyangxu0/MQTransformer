##MQTransformer
import torch.nn as nn
import torch
from einops import rearrange
from torch.nn.modules.utils import _pair as to_2tuple
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init
from timm.models.layers import trunc_normal_
import math
from einops import rearrange
from timm.models.layers import DropPath

from . import utils_heads
from .base import BaseHead

Norm = nn.LayerNorm

class MQFormerHead(BaseHead):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.enc_dec_times = 1
        self.head_endpoints = ['final']
        linear_dim = self.in_channels
        self.in_channels = 256
        out_channels = self.in_channels // 4

        self.bottleneck = nn.ModuleDict({t: utils_heads.ConvBNReLU(self.in_channels,
                                                                   out_channels,
                                                                   kernel_size=3,
                                                                   norm_layer=nn.BatchNorm2d,
                                                                   activation_layer=nn.ReLU)
                                         for t in self.tasks})
        self.final_logits = nn.ModuleDict({t: nn.Conv2d(out_channels,
                                                        self.task_channel_mapping[t]['final'],
                                                        kernel_size=1,
                                                        bias=True)
                                           for t in self.tasks})
        out_dim = 256
        self.parts_query_encoder_decoder_blocks = nn.ModuleList([
                                       parts_query_encoder_decoder(embed_dims=256,  num_parts=(64, 64, 64, 64), num_heads=(4, 8, 16, 32), drop_path_rate=0.5, task_num=len(self.tasks))
                                       for i in range(self.enc_dec_times)])

        self.patch_embed = PatchEmbed(in_channels=out_dim, embed_dims=out_dim, conv_type='Conv2d', kernel_size=4, stride=4, pad_to_patch_size=True, norm_layer=Norm)

        self.linears = nn.ModuleList([nn.Linear(linear_dim, self.in_channels)  for i in range(len(self.tasks)) ])
        self.init_weights()



    def forward(self, inp, inp_shape, **kwargs):
        inp = self._transform_inputs(inp)
        out = []
        for i, linear in enumerate(self.linears):
            x = linear(inp.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            out.append(x)

        x_patch = []
        for i in range(len(out)):
            x_patch.append(self.patch_embed(out[i]))



        # run times of encoder+decoder
        for i, parts_query_encoder_decoder_block in enumerate(self.parts_query_encoder_decoder_blocks):
            out = parts_query_encoder_decoder_block(out, x_patch)

        inp_dict = {t: out[idx] for idx, t in enumerate(self.tasks)}
        task_specific_feats = {t: self.bottleneck[t](inp_dict[t]) for t in self.tasks}

        final_pred = {t: self.final_logits[t](task_specific_feats[t]) for t in self.tasks}
        final_pred = {t: nn.functional.interpolate(final_pred[t], size=inp_shape, mode='bilinear', align_corners=False) for t in self.tasks}

        return {'final':final_pred}

class parts_query_encoder_decoder(nn.Module):
    def __init__(self,
                 inplanes=64,
                 pretrain_img_size=224,
                 in_channels=3,
                 embed_dims=256,
                 patch_size=4,
                 window_size=7,
                 mlp_ratio=4,
                 depths=(2, 2, 6, 2),
                 num_heads=(4, 8, 16, 32),
                 strides=(4, 2, 2, 2),
                 out_indices=(0, 1, 2, 3),
                 num_parts=(64, 64, 64, 64),
                 qkv_bias=True,
                 qk_scale=None,
                 patch_norm=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 task_num = None,
                 num_layers = 4,
                 use_abs_pos_embed=False,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 diff_patch_embeds=None):
        super(parts_query_encoder_decoder, self).__init__()

        self.act_layer = act_layer()
        self.task_num = task_num
        self.num_features = [int(embed_dims * 2 ** i) for i in range(num_layers)]

        self.rpn_tokens = nn.ParameterList([nn.Parameter(torch.Tensor(1, num_parts[0], embed_dims)) for i in range(self.task_num)])
        self.rpn_qpos = nn.ParameterList([nn.Parameter(torch.Tensor(1, num_parts[0], 1, embed_dims // num_heads[0])) for i in range(self.task_num)])
        self.rpn_kpos = nn.ParameterList([nn.Parameter(torch.Tensor(1, num_parts[0], 1, embed_dims // num_heads[0])) for i in range(self.task_num)])


        self.encoder1 = Encoder(embed_dims, num_parts=num_parts[0], num_enc_heads=num_heads[0], drop_path=drop_path_rate)
        self.decoder1 = Decoder(embed_dims, num_heads=num_heads[0], patch_size=patch_size, ffn_exp=3,drop_path=0.1)

        self.mhsa_1 = nn.MultiheadAttention(embed_dims, num_heads[0], dropout=0.1)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.norm_layer = norm_layer(embed_dims)
        self.ffn_mlp = Mlp(in_features=embed_dims, hidden_features=embed_dims, act_layer=act_layer, drop=0.)
        self._init_weights()
    def _init_weights(self):
        for i in range(self.task_num):
            init.kaiming_uniform_(self.rpn_tokens[i], a=math.sqrt(5))
            trunc_normal_(self.rpn_tokens[i], std=.02)

            init.kaiming_uniform_(self.rpn_qpos[i], a=math.sqrt(5))
            trunc_normal_(self.rpn_qpos[i], std=.02)

            init.kaiming_uniform_(self.rpn_kpos[i], a=math.sqrt(5))
            trunc_normal_(self.rpn_kpos[i], std=.02)


    def forward(self, x, y):
        x_1 = x[0]
        b,c,h,w = x[0].shape

        rpn_tokens_list = []
        rpn_qpos_list = []
        rpn_kpos_list = []
        for i, p in enumerate(self.rpn_tokens):
            _x = p.expand(x_1.shape[0], -1, -1)
            rpn_tokens_list.append(_x)

        for i, p in enumerate(self.rpn_qpos):
            _x = p.expand(x_1.shape[0], -1, -1, -1)
            rpn_qpos_list.append(_x)

        for i, p in enumerate(self.rpn_kpos):
            _x = p.expand(x_1.shape[0], -1, -1, -1)
            rpn_kpos_list.append(_x)


        x_list = []
        feats_list = []
        for i in range(len(x)):
            x_list.append(rearrange(x[i], "b c h w -> b (h w) c").contiguous())

        # encoder for part query
        part_for_dec_list = []
        for j in range(self.task_num):
            #parts = self.encoder1(x_list[j], parts=rpn_tokens_list_task4[j], qpos=rpn_qpos_1, mask=None)
            parts = self.encoder1(x_list[j], parts=rpn_tokens_list[j], qpos=rpn_qpos_list[j], mask=None)
            part_for_dec_list.append(parts)

        #parts self-attention
        part_cat = torch.cat(part_for_dec_list, dim=1).permute(1, 0, 2)
        part_cat = part_cat + self.drop_path(self.mhsa_1(part_cat, part_cat, part_cat)[0])
        part_cat = part_cat + self.drop_path(self.ffn_mlp(self.norm_layer(part_cat)))
        parts_cat = torch.split(part_cat.permute(1, 0, 2), 64, dim=1)

        #decoder
        for j in range(self.task_num):
            #feats = self.decoder1(x_list[j], parts_list=parts_cat[j], part_kpos=rpn_kpos_1, mask=None)[0]
            feats = self.decoder1(x_list[j], parts_list=parts_cat[j], part_kpos=rpn_kpos_list[j], mask=None)[0]
            feats= self.act_layer(feats) #feats[b,hw,c]
            feats = rearrange(feats, "b (h w) c -> b c h w", h=h, w=w)
            feats_list.append(feats)

        return feats_list


class Encoder(nn.Module):
    def __init__(self, dim, num_parts=64, num_enc_heads=1, drop_path=0.1, act=nn.GELU, has_ffn=True):
        super(Encoder, self).__init__()
        self.num_heads = num_enc_heads
        self.enc_attn = AnyAttention(dim, num_enc_heads)
        self.drop_path = DropPath(drop_prob=0.1) if drop_path else nn.Identity()
        self.reason = SimpleReasoning(num_parts, dim)
        self.enc_ffn = Mlp(dim, hidden_features=dim, act_layer=act) if has_ffn else None

    def forward(self, feats, parts=None, qpos=None, kpos=None, mask=None):
        """
        Args:
            feats: [B, patch_num * patch_size, C]
            parts: [B, N, C]
            qpos: [B, N, 1, C]
            kpos: [B, patch_num * patch_size, C]
            mask: [B, 1, patch_num, patch_size] if exists, else None
        Returns:
            parts: [B, N, C]
        """
        attn_out = self.enc_attn(q=parts, k=feats, v=feats, qpos=qpos, kpos=kpos, mask=mask)
        parts = parts + self.drop_path(attn_out)
        parts = self.reason(parts)
        if self.enc_ffn is not None:
            parts = parts + self.drop_path(self.enc_ffn(parts))
        return parts

class Decoder(nn.Module):
    def __init__(self, dim, num_heads=8, patch_size=7, ffn_exp=3, act=nn.GELU, drop_path=0.1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.attn1 = AnyAttention(dim, num_heads)
        self.attn2 = AnyAttention(dim, num_heads)
        self.rel_pos = FullRelPos(patch_size, patch_size, dim // num_heads)
        self.ffn1 = Mlp(dim, hidden_features=dim * ffn_exp, act_layer=act, norm_layer=Norm)
        self.ffn2 = Mlp(dim, hidden_features=dim * ffn_exp, act_layer=act, norm_layer=Norm)
        self.drop_path = DropPath(drop_path)
        self.adapool2d = nn.AdaptiveAvgPool2d((1, 1))
        self.mhsa = nn.MultiheadAttention(dim, num_heads)

    def forward(self, x, parts_list=None, part_kpos=None, mask=None, P=0):
        """
        Args:
            x: [B, patch_num * patch_size, C]
            parts: [B, N, C]
            part_kpos: [B, N, 1, C]
            mask: [B, 1, patch_num, patch_size] if exists, else None
            P: patch_num
        Returns:
            feat: [B, patch_num, patch_size, C]
        """
        outs_list = []
        task = 1
        dec_mask = None if mask is None else rearrange(mask.squeeze(1), "b h w -> b (h w) 1 1")
        for i in range(task):
            parts = parts_list[i] if isinstance(parts_list[i], (list)) else parts_list
            #x = x+out if i >0 else x
            out = self.attn1(q=x, k=parts, v=parts, kpos=part_kpos, mask=dec_mask)
            out = x + self.drop_path(out)
            out = out + self.drop_path(self.ffn1(out))

            #outs = rearrange(out, "b (p k) c -> (b p) k c", p=P)
            local_out = self.mhsa(out, out, out)[0]
            #local_out = self.attn2(q=out, k=out, v=out, mask=mask, rel_pos=None)  #rel_pos=self.rel_pos
            outs = x + self.drop_path(local_out)
            outs = outs + self.drop_path(self.ffn2(outs))
            #outs = rearrange(outs, "(b p) k c -> b p k c", p=P)
            outs_list.append(outs)

        # outs_0 = self.adapool2d(outs_list[0])
        # outs_1 = self.adapool2d(outs_list[1])
        # outs_2 = self.adapool2d(outs_list[2])
        # outs_3 = self.adapool2d(outs_list[3])
        # outs = outs + outs_0 + outs_1 + outs_2 + outs_3
        return outs_list

class AnyAttention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False):
        super(AnyAttention, self).__init__()
        self.norm_q, self.norm_k, self.norm_v = Norm(dim), Norm(dim), Norm(dim)
        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias)

        self.scale = (dim / num_heads) ** (-0.5)
        self.num_heads = num_heads
        self.proj = nn.Linear(dim, dim)

    def get_qkv(self, q, k, v, qpos, kpos):
        q = apply_pos(q, qpos, self.num_heads)
        k = apply_pos(k, kpos, self.num_heads)
        v = apply_pos(v, None, 0)
        q, k, v = self.norm_q(q), self.norm_k(k), self.norm_v(v)
        q, k, v = self.to_q(q), self.to_k(k), self.to_v(v)
        return q, k, v

    def forward(self, q=None, k=None, v=None, qpos=None, kpos=None, mask=None, rel_pos=None):
        q, k, v = self.get_qkv(q, k, v, qpos, kpos)

        # reshape
        q = rearrange(q, "b n (g c) -> b n g c", g=self.num_heads)
        k = rearrange(k, "b n (g c) -> b n g c", g=self.num_heads)
        v = rearrange(v, "b n (g c) -> b n g c", g=self.num_heads)

        # attn matrix calculation
        attn = torch.einsum("b q g c, b k g c -> b q g k", q, k)
        if rel_pos is not None:
            attn = rel_pos(q, attn)
        attn *= self.scale
        if mask is not None:
            attn = attn.masked_fill(mask.bool(), value=float('-inf'))
        attn = F.softmax(attn, dim=-1)
        if mask is not None:
            attn = attn.masked_fill(mask.bool(), value=0)
        out = torch.einsum("b q g k, b k g c -> b q g c", attn, v.float())
        out = rearrange(out, "b q g c -> b q (g c)")
        out = self.proj(out)
        return out

def apply_pos(tensor, pos, num_heads):
    if pos is None:
        return tensor
    elif len(tensor.shape) != len(pos.shape):
        tensor = rearrange(tensor, "b n (g c) -> b n g c", g=num_heads)
        tensor = tensor + pos
        tensor = rearrange(tensor, "b n g c -> b n (g c)")
    else:
        tensor = tensor + pos
    return tensor

class SimpleReasoning(nn.Module):
    def __init__(self, np, dim):
        super(SimpleReasoning, self).__init__()
        self.norm = Norm(dim)
        self.linear = nn.Conv1d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        tokens = self.norm(x).permute(0,2,1)
        tokens = self.linear(tokens).permute(0,2,1)
        return x + tokens

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = int(hidden_features) or in_features
        self.norm = norm_layer(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class FullRelPos(nn.Module):
    def __init__(self, h, w, dim, drop_ratio=0.):
        super(FullRelPos, self).__init__()
        self.h, self.w = h, w
        self.rel_emb_h = nn.Parameter(torch.Tensor(2 * h - 1, dim // 2))  # [-(q-1), q-1]
        self.rel_emb_w = nn.Parameter(torch.Tensor(2 * w - 1, dim // 2))  # [-(q-1), q-1]

        # get relative coordinates of the q-k index table
        coords_h = torch.arange(h)
        coords_w = torch.arange(w)
        self.rel_idx_h = coords_h[None, :] - coords_h[:, None]
        self.rel_idx_w = coords_w[None, :] - coords_w[:, None]
        self.rel_idx_h += h - 1
        self.rel_idx_w += w - 1

        nn.init.normal_(self.rel_emb_w, std=dim ** -0.5)
        nn.init.normal_(self.rel_emb_h, std=dim ** -0.5)
        trunc_normal_(self.rel_emb_w, std=.02)
        trunc_normal_(self.rel_emb_h, std=.02)
        self.drop_ratio = drop_ratio

    def forward(self, q, attn):
        abs_pos_h = self.rel_emb_h[self.rel_idx_h.view(-1)]
        abs_pos_w = self.rel_emb_w[self.rel_idx_w.view(-1)]
        abs_pos_h = rearrange(abs_pos_h, "(q k) c -> q k c", q=self.h)  # [qh, kh, c]
        abs_pos_w = rearrange(abs_pos_w, "(q k) c -> q k c", q=self.w)  # [qw, kw, c]

        q = rearrange(q, "b (qh qw) g (n c) -> b qh qw g n c", qh=self.h, qw=self.w, n=2)
        logits_h = torch.einsum("b h w g c, h k c -> b h w g k", q[..., 0, :], abs_pos_h)
        logits_w = torch.einsum("b h w g c, w k c -> b h w g k", q[..., 1, :], abs_pos_w)
        logits_h = rearrange(logits_h, "b h w g k -> b (h w) g k 1")
        logits_w = rearrange(logits_w, "b h w g k -> b (h w) g 1 k")

        attn = rearrange(attn, "b q g (kh kw) -> b q g kh kw", kh=self.h, kw=self.w)
        attn += logits_h
        attn += logits_w
        return rearrange(attn, "b q g h w -> b q g (h w)")




class PatchEmbed(nn.Module):
    def __init__(self,
                 in_channels=3,
                 embed_dims=768,
                 conv_type=None,
                 kernel_size=16,
                 stride=16,
                 padding=0,
                 dilation=1,
                 pad_to_patch_size=True,
                 norm_layer=None):
        super(PatchEmbed, self).__init__()

        self.embed_dims = embed_dims
        if stride is None:
            stride = kernel_size
        self.pad_to_patch_size = pad_to_patch_size

        # The default setting of patch size is equal to kernel size.
        patch_size = kernel_size
        if isinstance(patch_size, int):
            patch_size = to_2tuple(patch_size)
        elif isinstance(patch_size, tuple):
            if len(patch_size) == 1:
                patch_size = to_2tuple(patch_size[0])
            assert len(patch_size) == 2, \
                f'The size of patch should have length 1 or 2, ' \
                f'but got {len(patch_size)}'

        self.patch_size = patch_size
        # Use conv layer to embed
        #self.projection = nn.Unfold(kernel_size=(14, 14), stride=(8, 8), padding=(3, 3))
        conv_type = conv_type or 'Conv2d'
        if conv_type == 'Conv2d':
            self.projection = nn.Conv2d(
                in_channels=in_channels,
                out_channels=embed_dims,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation
            )
        else:
            raise NotImplementedError
        if norm_layer is not None:
            self.norm = norm_layer(embed_dims)
        else:
            self.norm = None

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]
        # TODO: Process overlapping op
        if self.pad_to_patch_size:
            # Modify H, W to multiple of patch size.
            if H % self.patch_size[0] != 0:
                x = F.pad(
                    x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))
            if W % self.patch_size[1] != 0:
                x = F.pad(
                    x, (0, self.patch_size[1] - W % self.patch_size[1], 0, 0))
        x = self.projection(x)
        self.DH, self.DW = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=self.DH, w=self.DW)
        return x

