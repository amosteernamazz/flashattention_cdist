import torch
from numpy import sqrt
import numpy as np
# Initialization of params
torch.random.manual_seed(0)
batch_size = 2
seq_len = 256*10

nheads = 3
dim = 128
dtype = torch.float
scale = 1.0 * dim ** -0.5
dropout_p = 0
softmax_scale = scale
causal = False; return_softmax=False
num_splits=0; generator=None
print(f"\nbatch_size {batch_size}, seq_len {seq_len}, nheads {nheads}, dim {dim} \n")

# def init_index_tensor(rows, columns, dtype):
#     tensor = torch.arange(0, rows).cuda().unsqueeze(1).expand(-1, columns) + \
#         torch.round(torch.arange(1, columns+1) / 100.0, decimals = 2).cuda().unsqueeze(0).expand(rows, -1)
    
#     return tensor.to(dtype)

##  backward of torch 
# q = init_index_tensor(batch_size*seq_len, nheads*dim, dtype)
# q= torch.reshape(q,(batch_size, seq_len, 1, nheads, dim))
# q_torch = torch.tensor(q, dtype=dtype,requires_grad=True, device='cuda:0')
# q_torch = torch.clamp(q_torch, 0, 1)
q_torch = torch.randn(batch_size, seq_len, 1, nheads, dim, dtype = dtype, requires_grad=True).cuda()
# q = torch.clamp(q, 0, 1)
# k= init_index_tensor(batch_size*seq_len, nheads*dim, dtype)
# k = torch.reshape(k,(batch_size, seq_len, 1, nheads, dim))
# k_torch = torch.tensor(k, dtype=dtype,requires_grad=True, device='cuda:0')
# k_torch = torch.clamp(k_torch, 0, 1)

k_torch = torch.randn(batch_size, seq_len, 1, nheads, dim, dtype = dtype, requires_grad=True).cuda()

# k_torch = k_torch * 10

v_torch = torch.randn([batch_size, seq_len, 1, nheads, dim], dtype=dtype, requires_grad=True).cuda()

q_torch.retain_grad(); k_torch.retain_grad(); v_torch.retain_grad()
qkv_torch = torch.cat([q_torch,k_torch,v_torch], -1).reshape(batch_size, seq_len, nheads, 3*dim)
qkv = qkv_torch.permute(0, 2, 1, 3)
qkv = qkv.reshape(batch_size*nheads, seq_len, 3*dim)
q, k, v = qkv.chunk(3, dim=2)
cdist_torch = - torch.cdist(q.to(torch.float32), k.to(torch.float32), p=2).to(dtype)
ds = 1 - cdist_torch
cdist_torch.backward(ds)
dq_torch = q_torch.grad; dk_torch = k_torch.grad; dv_torch = v_torch.grad
# print(cdist_torch.size())
#### Backward of flattn
q_flattn = torch.squeeze(q_torch.clone(), dim=2)
k_flattn = torch.squeeze(k_torch.clone(), dim=2)
v_flattn = torch.squeeze(v_torch.clone(), dim=2)

dq_flattn = torch.zeros(q_flattn.size()).cuda(); dk_flattn = torch.zeros(k_flattn.size()).cuda()

# A_trans = dq_flattn.reshape(batch_size, nheads, dim, seq_len)
# print(dq_flattn.size())
br = 8
bc = 16

for batch in range(batch_size):
    for head in range(nheads):
        scores = cdist_torch[batch*nheads+head, :, :]
        dscores = ds[batch*nheads+head, :, :]
        sprime = dscores/scores
        sprimeT = sprime.T
        for j in range(seq_len // bc):
            Kj = k_flattn[batch,j*bc:(j+1)*bc,head,:]
            Vj = v_flattn[batch,j*bc:(j+1)*bc,head,:]
            # print("vj size")
            # print(Vj.size())
            for i in range(seq_len // br):
                Qi = q_flattn[batch,i*br:(i+1)*br,head,:]

                dq_flattn[batch, i*br:(i+1)*br, head, :] = dq_flattn[batch, i*br:(i+1)*br, head, :] + \
                torch.einsum("rd,r ->rd", Qi , sprime[i*br:(i+1)*br,j*bc:(j+1)*bc].sum(axis = -1))\
                -torch.einsum("rc, cd->rd",sprime[i*br:(i+1)*br,j*bc:(j+1)*bc], Kj)

                # dk_flattn[batch, j*bc:(j+1)*bc, head, :] = dk_flattn[batch, j*bc:(j+1)*bc, head, :] + \
                # torch.einsum("rd,r ->rd", Kj , sprimeT[j*bc:(j+1)*bc,i*br:(i+1)*br].sum(axis = -1))\
                # -torch.einsum("rc, cd->rd",sprimeT[j*bc:(j+1)*bc,i*br:(i+1)*br], Qi)

                dk_flattn[batch, j*bc:(j+1)*bc, head, :] = dk_flattn[batch, j*bc:(j+1)*bc, head, :] + \
                torch.einsum("rd,r ->rd", Kj , sprime[i*br:(i+1)*br,j*bc:(j+1)*bc].sum(axis = 0)) \
                -torch.einsum("rc, cd->rd",sprime[i*br:(i+1)*br,j*bc:(j+1)*bc].T, Qi)

import pdb; pdb.set_trace()
print(f"dq diff", (torch.squeeze(dq_torch, dim=2) - dq_flattn).abs().max())
print(f"dk diff", (torch.squeeze(dk_torch, dim=2) - dk_flattn).abs().max())

