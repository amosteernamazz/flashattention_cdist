import torch
import torch.nn as nn
import time

def init_index_tensor(rows, columns, dtype):
    tensor = torch.arange(0, rows).cuda().unsqueeze(1).expand(-1, columns) + \
        torch.round(torch.arange(1, columns+1) / 100.0, decimals = 2).cuda().unsqueeze(0).expand(rows, -1)
    
    return tensor.to(dtype)

def print_scores(scores, threadIdx = 0, blockIdx = (0,0,0)):
    torch.set_printoptions(sci_mode=True, precision = 2)

    seq = blockIdx[0]; batch = blockIdx[1]; nhead = blockIdx[2]
    warpId = int(threadIdx/32)
    seq_len = scores.shape[1]
            
    for iter, nblock in enumerate(range(int(seq_len/128)-1, -1, -1)):
        print(f"\n\nTorch: scores of block {blockIdx}, thread {threadIdx}, nblock {nblock}")
        seq_offset_k = []
        for j in range(16):
            seq_offset_k.append(8*j + 128 *nblock + int(threadIdx%4)*2)
            seq_offset_k.append(8*j + 128 *nblock + int(threadIdx%4)*2 +1)

        seq_offset_q = int(threadIdx%32/4) + 16*warpId + seq*128
        print(scores[batch][seq_offset_q][nhead][seq_offset_k])
        
        seq_offset_q = int(threadIdx%32/4) + 16*warpId + 8 + seq*128
        print(scores[batch][seq_offset_q][nhead][seq_offset_k])
        
        seq_offset_q = int(threadIdx%32/4) + 16*warpId + 16*4 + seq*128
        print(scores[batch][seq_offset_q][nhead][seq_offset_k])
        
        seq_offset_q = int(threadIdx%32/4) + 16*warpId + 16*4 + 8 + seq*128
        print(scores[batch][seq_offset_q][nhead][seq_offset_k])
                
    torch.set_printoptions(sci_mode=False)    


torch.random.manual_seed(0)
batch_size = 2
seq_len = 256*10

nheads = 3
dim = 128
dtype = torch.bfloat16
scale = 1.0 * dim ** -0.5
dropout_p = 0
softmax_scale = scale
causal = False; return_softmax=False
num_splits=0; generator=None

print(f"\nbatch_size {batch_size}, seq_len {seq_len}, nheads {nheads}, dim {dim} \n")

# q = init_index_tensor(batch_size*seq_len, nheads*dim, dtype)
# q= torch.reshape(q,(batch_size, seq_len, 1, nheads, dim))
# q_torch = torch.tensor(q, dtype=dtype,requires_grad=True, device='cuda:0')
# q_torch = q_torch * 0.001
# print("q_torch: ");print(q_torch)
# q = init_index_tensor(batch_size*seq_len, nheads*dim, dtype)
q_torch = torch.randn(batch_size, seq_len, 1, nheads, dim, dtype = dtype, requires_grad=True).cuda()
# print(q_torch)
q_flattn = torch.squeeze(q_torch, dim=2)


# k= init_index_tensor(batch_size*seq_len, nheads*dim, dtype)
# k = torch.reshape(k,(batch_size, seq_len, 1, nheads, dim))
# k_torch = torch.tensor(k, dtype=dtype,requires_grad=True, device='cuda:0')
# k_torch = k_torch * 0.001
# print("k_torch: ");print(k_torch)

# k= init_index_tensor(batch_size*seq_len, nheads*dim, dtype)
k_torch = torch.randn(batch_size, seq_len, 1, nheads, dim, dtype = dtype, requires_grad=True).cuda()
k_flattn = torch.squeeze(k_torch, dim=2)

v_torch = torch.randn([batch_size, seq_len, 1, nheads, dim], dtype=dtype, requires_grad=True).cuda()
v_flattn = torch.squeeze(v_torch, dim=2)


q_torch.retain_grad(); k_torch.retain_grad(); v_torch.retain_grad()
qkv_torch = torch.cat([q_torch,k_torch,v_torch], -1).reshape(batch_size, seq_len, nheads, 3*dim) 


## Initial backward
dq_flattn, dk_flattn, dv_flattn = torch.empty_like(q_flattn), torch.empty_like(k_flattn), torch.empty_like(v_flattn)

use_flattn = 1; use_torch= 1; use_cdist = 1


if (use_flattn):
    import flash_attn_2_cuda as flash_attn_cuda
    torch_allo_1 = torch.cuda.memory_allocated(); torch_reserve_1 = torch.cuda.memory_reserved()
    if use_cdist:
        seq_len_padding = (int((seq_len + 127) / 128) ) *128
        q_square_padded = torch.zeros(batch_size, nheads, seq_len_padding, dtype = dtype).cuda()
        k_square_padded = torch.zeros(batch_size, nheads, seq_len_padding, dtype = dtype).cuda()
        q_square_padded[:,:, :seq_len] = torch.mul(q_flattn, q_flattn).sum(axis = -1).permute(0, 2, 1).contiguous()
        k_square_padded[:,:, :seq_len] = torch.mul(k_flattn, k_flattn).sum(axis = -1).permute(0, 2, 1).contiguous()
    for i in range(1):
        
        torch.cuda.synchronize()
        t0 = time.time()
        if use_cdist:
            flash_out, q, k, v, out_padded, softmax_lse, S_dmask = flash_attn_cuda.fwd(
            q_flattn, k_flattn, v_flattn, None, dropout_p, softmax_scale, causal, return_softmax, None, True, q_square_padded, k_square_padded)
            dout_flattn = (1 - flash_out)
            dq_flattn, dk_flattn, dv_flattn, softmax_d, = flash_attn_cuda.bwd(
                dout_flattn, q_flattn, k_flattn, v_flattn, flash_out, softmax_lse, dq_flattn, dk_flattn, dv_flattn, dropout_p, softmax_scale, causal, None, True, q_square_padded, k_square_padded)
            # a = q_flattn.to(torch.float32).permute(0, 2, 1, 3).reshape(batch_size*nheads, seq_len, dim)
            # b = k_flattn.to(torch.float32).permute(0, 2, 1, 3).reshape(batch_size*nheads, seq_len, dim)
            # cdist_flattn = torch.cdist(a, b, p=2)\
            #                         .to(torch.float16).reshape(batch_size, nheads, seq_len, seq_len).permute(0, 2, 1, 3)
            # import pdb; pdb.set_trace()
            # dq_flattn = 0
            # dq_flattn = softmax_d*(2*q_flattn - 2*k_flattn); 
            # dk_flattn = 0; 

        else:
            flash_out, q_flattn, k_flattn, v_flattn, out_padded, softmax_lse, S_dmask = flash_attn_cuda.fwd(
            q_flattn, k_flattn, v_flattn, None, dropout_p, softmax_scale, causal, return_softmax, None, False, None, None)
            dout_flattn = (1 - flash_out)
            dq_flattn, dk_flattn, dv_flattn, softmax_d, = flash_attn_cuda.bwd(
                dout_flattn,q_flattn, k_flattn, v_flattn, flash_out, softmax_lse, dq_flattn, dk_flattn, dv_flattn, dropout_p, softmax_scale, causal, None, False, None, None
            )
        torch.cuda.synchronize()
        t1 = time.time()

    torch_allo_2 = torch.cuda.memory_allocated(); torch_reserve_2 = torch.cuda.memory_reserved()
    torch_allocated = (torch_allo_2-torch_allo_1)/1024/1024
    torch_reserved = (torch_reserve_2-torch_reserve_1)/1024/1024
    print(f"flattn allocates {torch_allocated} MB")
    print(f"flattn reserves {torch_reserved} MB")
        

class QKVAttention(nn.Module):
    """
    QKV attention module that support similarity computed using dot product and

    L2 norm.
    """
    def __init__(self,
                 n_heads,
                 scale,
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
        qkv = qkv.permute(0, 2, 1, 3)
        qkv = qkv.reshape(batch_size*nheads, seq_len, 3*dim)
        q, k, v = qkv.chunk(3, dim=2)

        if self.scale_in:
            q = q * self.scale
            k = k * self.scale

        if self.l2_attention:
            attn_logits = - torch.cdist(q.to(torch.float32), k.to(torch.float32), p=2).to(dtype)
            
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
        a = a.reshape(batch_size, nheads, seq_len, dim).permute(0, 2, 1, 3)
        return a
    
if (use_torch):
    torch_allo_1 = torch.cuda.memory_allocated(); torch_reserve_1 = torch.cuda.memory_reserved()
    if use_cdist:
        attn = QKVAttention(n_heads=nheads, scale=scale, l2_attention=True, scale_in=False).cuda()
    else:
        attn = QKVAttention(n_heads=nheads, scale=scale, l2_attention=False, scale_in=False).cuda()

    for i in range(10):
        torch.cuda.synchronize()
        t2 = time.time()
        attn_out = attn(qkv_torch)
        torch.cuda.synchronize()
        t3 = time.time()
    
    attn_out.backward(1-attn_out)
    dq_torch = torch.squeeze(q_torch.grad, dim=2); dk_torch = torch.squeeze(k_torch.grad, dim=2); dv_torch = torch.squeeze(v_torch.grad, dim=2)
    torch_allo_2 = torch.cuda.memory_allocated(); torch_reserve_2 = torch.cuda.memory_reserved()
    torch_allocated = (torch_allo_2-torch_allo_1)/1024/1024
    torch_reserved = (torch_reserve_2-torch_reserve_1)/1024/1024
    print(f"torch allocates {torch_allocated} MB")
    print(f"torch reserves {torch_reserved} MB")

if use_torch and use_flattn:
    
    diff_fwd = (flash_out - attn_out).abs()
    # import pdb; pdb.set_trace()
    diff_bwd_dq = (dq_flattn - dq_torch).abs()
    diff_bwd_dk = (dk_flattn - dk_torch).abs()
    diff_bwd_dv = (dv_flattn - dv_torch).abs()
    print(f'flash attention2 time: {t1-t0:.6f}\ts')
    print(f'maual attention time: {t3-t2:.6f}\ts')
    print(f'forward absmax diff: {diff_fwd.max()}\n')
    print(f'diff_bwd_dq: {diff_bwd_dq.max()}\n')
    print(f'diff_bwd_dk: {diff_bwd_dk.max()}\n')
    print(f'diff_bwd_dv: {diff_bwd_dv.max()}\n')
