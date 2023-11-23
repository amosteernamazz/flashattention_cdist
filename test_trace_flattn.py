import torch
import torch.nn as nn
import time
import flash_attn_2_cuda as flash_attn_cuda




def init_index_tensor(rows, columns):
    tensor = torch.arange(0, rows).cuda().unsqueeze(1).expand(-1, columns) + \
        torch.round(torch.arange(1, columns+1) / 100.0, decimals = 2).cuda().unsqueeze(0).expand(rows, -1)
    return tensor.half()

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


# 设置打印格式
torch.set_printoptions(sci_mode=False)
# print(f"flash_attn_cuda.__path__: {flash_attn_cuda.__path__}")
torch.random.manual_seed(0)
batch_size = 1
seq_len = 256*10; 
nheads = 1
dim=32
dtype = torch.float16
scale = 1.0 * dim ** -0.5

print(f"\nbatch_size {batch_size}, seq_len {seq_len}, nheads {nheads}, dim {dim} \n")


# q = init_index_tensor(batch_size*seq_len, nheads*dim)
q = torch.randn(batch_size*seq_len, nheads*dim).half()
q = torch.reshape(q, [batch_size, seq_len, 1, nheads, dim]).cuda(); q2 = torch.squeeze(q, dim=2)


# k= init_index_tensor(batch_size*seq_len, nheads*dim)
k =  torch.randn(batch_size*seq_len, nheads*dim).half()
k = torch.reshape(k, [batch_size, seq_len, 1, nheads, dim]).cuda(); k2 = torch.squeeze(k, dim=2)

v = torch.randn([batch_size, seq_len, 1, nheads, dim], dtype=dtype).cuda(); v2 = torch.squeeze(v, dim=2)
qkv = torch.cat([q,k,v], 2)

a = q2.to(torch.float32).permute(0, 2, 1, 3).reshape(batch_size*nheads, seq_len, dim)
b = k2.to(torch.float32).permute(0, 2, 1, 3).reshape(batch_size*nheads, seq_len, dim)
cdist_batch = torch.cdist(a, b, p=2)\
                        .to(torch.float16).reshape(batch_size, nheads, seq_len, seq_len).permute(0, 2, 1, 3)

cdist_batch.permute(0, 2, 1, 3)
scores = torch.zeros([batch_size, nheads, seq_len, seq_len]).half().cuda()
use_flash = 1; use_torch= 1


# torch.round(q2, decimals = 2)
dropout_p = 0
softmax_scale = scale
causal = False; return_softmax=False
num_splits=0; generator=None


if (use_flash):
    q_square = torch.mul(q2, q2).sum(axis = -1).permute(0, 2, 1).contiguous()
    k_square = torch.mul(k2, k2).sum(axis = -1).permute(0, 2, 1).contiguous()
    for i in range(1):
        flash_out, q, k, v, out_padded, softmax_lse, S_dmask = flash_attn_cuda.fwd(
        q2, k2, v2, None, dropout_p, softmax_scale, causal, return_softmax, None, True, q_square, k_square, scores)
        diff = cdist_batch.permute(0, 2, 1, 3) + scores
        # import pdb; pdb.set_trace()



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
        q, k, v = qkv.chunk(3, dim=2)

        if self.scale_in:
            q = q * self.scale
            k = k * self.scale

        if self.l2_attention:
            attn_logits = - torch.cdist(q.to(torch.float32), k.to(torch.float32), p=2).to(torch.float16)
            
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
if (use_torch):
    with torch.no_grad():
        torch_allo_1 = torch.cuda.memory_allocated(); torch_reserve_1 = torch.cuda.memory_reserved()
        attn = QKVAttention(n_heads=nheads, scale=scale, l2_attention=True, scale_in=False).cuda()

        qkv = qkv.permute(0, 3, 1, 2, 4)
        qkv = qkv.reshape(batch_size*nheads, seq_len, 3*dim)

        for i in range(10):
            torch.cuda.synchronize()
            t2 = time.time()
            attn_out= attn(qkv)
            torch.cuda.synchronize()
            t3 = time.time()
        attn_out = attn_out.reshape(batch_size, nheads, seq_len, dim).permute(0, 2, 1, 3)
        torch_allo_2 = torch.cuda.memory_allocated(); torch_reserve_2 = torch.cuda.memory_reserved()
        torch_allocated = (torch_allo_2-torch_allo_1)/1024/1024
        torch_reserved = (torch_reserve_2-torch_reserve_1)/1024/1024
        print(f"torch allocates {torch_allocated} MB")
        print(f"torch reserves {torch_reserved} MB")

if use_torch and use_flash:
    
    diff_fwd = (flash_out - attn_out).abs()

    # print(f'flash attention2 time: {t1-t0:.6f}\ts')
    # print(f'maual attention time: {t3-t2:.6f}\ts')
    print(f'forward absmax diff: {diff_fwd.max()}\n')

