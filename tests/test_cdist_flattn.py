

import torch
import torch.nn as nn

import pytest

from einops import rearrange, repeat
import flash_attn_2_cuda as flash_attn_cuda

class QKVAttention(nn.Module):
    """
    QKV attention module that support similarity computed using dot product and

    L2 norm.
    """
    def __init__(self,
                 n_heads,
                 scale,
                 dtype,
                 l2_attention=True,
                 scale_in=True,
                 mask_self=False,
                 mask_value='-inf'):
        """Initializes with layer settings.

        Args:
            n_head: Number of heads for the multi-head attention.
            scale: The scale for the attention.
            l2_attention: Whether or not use l2 distance to compute similarity.
            mask_self: Whether to omit the attention to self.
            mask_value: Default value to fill in the distance matrix.
        """
        super().__init__()
        self.n_heads = n_heads
        self.scale = scale
        self.l2_attention = l2_attention
        self.scale_in = scale_in
        self.mask_value = mask_value
        self.mask_self = mask_self
        self.dtype = dtype

    def extra_repr(self) -> str:
        return (f'num_heads={self.n_heads}, '
                f'scale={self.scale}, '
                f'mask_val={self.mask_value}')

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (HW) x C*3] tensor of Qs, Ks, and Vs.
        :return: an [N x (HW) x C] tensor after attention.
        """
        q, k, v = qkv.chunk(3, dim=2)

        if self.scale_in:
            q = q * self.scale
            k = k * self.scale

        if self.l2_attention:
            # attn_logits = - torch.cdist(q.to(torch.float32), k.to(torch.float32), p=2).to(dtype)
            attn_logits = - torch.cdist(q, k, p=2).to(self.dtype)
            
            if self.mask_self:
                mask = torch.eye(attn_logits.shape[1],
                                 device=attn_logits.device,
                                 dtype=torch.bool)
                attn_logits = attn_logits.masked_fill(mask,
                                                      float(self.mask_value))
        else:
            attn_logits = torch.einsum('b t c, b s c -> b t s', q, k)

        if not self.scale_in:
            attn_logits = attn_logits * self.scale

        attn = torch.softmax(attn_logits, dim=-1)
        #a = torch.bmm(attn, v)
        a = torch.einsum('b t s, b s c -> b t c', attn, v)
        
        return a
    

# @pytest.mark.parametrize('dtype', ([torch.float16] if is_sm75 else [torch.float16, torch.bfloat16]))
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('causal', [False])
@pytest.mark.parametrize('dim', [32, 64, 128, 160, 192, 256])
@pytest.mark.parametrize('seq_len', [128, 256, 384, 512, 1024, 2048])
@pytest.mark.parametrize('batch_size', [1, 2, 3])
@pytest.mark.parametrize('nheads', [1, 2, 3])
@pytest.mark.parametrize('dropout_p', [0.0])
# @pytest.mark.parametrize('dim', [32])
# @pytest.mark.parametrize('seq_len', [128])
# @pytest.mark.parametrize('batch_size', [1])
# @pytest.mark.parametrize('nheads', [1])


def test_flash_attn_cdist(batch_size, seq_len, nheads, dim, dropout_p, causal, dtype):
    scale = 1.0 * dim ** -0.5; softmax_scale = scale; return_softmax=False

    q_torch = torch.randn(batch_size, seq_len, 1, nheads, dim, dtype = dtype).cuda()
    q_flattn = torch.squeeze(q_torch, dim=2)

    # k= init_index_tensor(batch_size*seq_len, nheads*dim, dtype)
    k_torch = torch.randn(batch_size, seq_len, 1, nheads, dim, dtype = dtype).cuda()
    k_flattn = torch.squeeze(k_torch, dim=2)

    v_torch = torch.randn([batch_size, seq_len, 1, nheads, dim], dtype=dtype).cuda()
    v_flattn = torch.squeeze(v_torch, dim=2)

    ###### FLASH ATTENTION ####################
    seq_len_padding = (int((seq_len + 127) / 128) ) *128
    q_square_padded = torch.zeros(batch_size, nheads, seq_len_padding, dtype = dtype).cuda()
    k_square_padded = torch.zeros(batch_size, nheads, seq_len_padding, dtype = dtype).cuda()
    q_square_padded[:,:, :seq_len] = torch.mul(q_flattn, q_flattn).sum(axis = -1).permute(0, 2, 1).contiguous()
    k_square_padded[:,:, :seq_len] = torch.mul(k_flattn, k_flattn).sum(axis = -1).permute(0, 2, 1).contiguous()
    flash_out, q, k, v, out_padded, softmax_lse, S_dmask = flash_attn_cuda.fwd(
            q_flattn, k_flattn, v_flattn, None, dropout_p, softmax_scale, causal, return_softmax, None, True, q_square_padded, k_square_padded)
    #### TORCH ATTENTION #########################
    attn = QKVAttention(n_heads=nheads, scale=scale, dtype=dtype, l2_attention=True, scale_in=False).cuda()
    
    qkv_torch = torch.cat([q_torch,k_torch,v_torch], 2)
    qkv = qkv_torch.permute(0, 3, 1, 2, 4)
    qkv = qkv.reshape(batch_size*nheads, seq_len, 3*dim)
    attn_out= attn(qkv)
    attn_out = attn_out.reshape(batch_size, nheads, seq_len, dim).permute(0, 2, 1, 3)

    assert torch.allclose(flash_out, attn_out, rtol=1e-03, atol=1e-02, equal_nan=True)

   