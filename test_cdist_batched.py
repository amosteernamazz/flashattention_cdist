import torch
import torch.nn as nn
import time

batch_size = 64
seq_len = 128; print(f"========================\nseq_len {seq_len}")
nheads = 4
dim=32
dtype = torch.float16

q_batch = torch.randn([batch_size, seq_len, 1, nheads, dim], dtype=dtype).cuda(); q_batch = torch.squeeze(q_batch, dim=2)
k_batch = torch.randn([batch_size, seq_len, 1, nheads, dim], dtype=dtype).cuda(); k_batch = torch.squeeze(k_batch, dim=2)

a = q_batch.to(torch.float32).permute(0, 2, 1, 3).reshape(batch_size*nheads, seq_len, dim)
b = k_batch.to(torch.float32).permute(0, 2, 1, 3).reshape(batch_size*nheads, seq_len, dim)
cdist_batch = torch.cdist(a, b, p=2)\
                        .to(torch.float16).reshape(batch_size, nheads, seq_len, seq_len).permute(0, 2, 1, 3)

q_batch = torch.randn([batch_size, seq_len, nheads, dim], dtype=dtype).cuda()
Q_2_batch = torch.mul(q_batch, q_batch).sum(axis = -1)
K_2_batch = torch.mul(k_batch, k_batch).sum(axis = -1)
Q_2_batch.unsqueeze(3).expand(-1, -1, -1, dim)
import pdb; pdb.set_trace()
expand_express_batch = (Q_2_batch.unsqueeze(3).expand(-1, -1, -1, seq_len) + \
                  K_2_batch.unsqueeze(3).expand(-1, -1, -1, seq_len).permute(0,3,2,1) - 2*torch.matmul\
                    (q_batch.permute(0, 2, 1, 3), k_batch.permute(0, 2, 3, 1)).permute(0, 2, 1, 3)\
                        ).sqrt()

abs_diff_batch = (cdist_batch - expand_express_batch).abs().max()

print(f"Max difference of batch cdist is {abs_diff_batch}")

# ### Non-batched
# Q = torch.squeeze(torch.squeeze(q_batch, dim=0), dim=1)
# K = torch.squeeze(torch.squeeze(k_batch, dim=0), dim = 1)

# cdist = torch.cdist(Q.float(), K.float(), p = 2).half()

# Q_2 = torch.mul(Q, Q).sum(axis = 1)
# K_2 = torch.mul(K, K).sum(axis = 1)
# import pdb; pdb.set_trace()
# expand_express = (Q_2.repeat(seq_len, 1).T + K_2.repeat(seq_len,1) - 2*torch.matmul(Q, K.T)).sqrt()
# import pdb; pdb.set_trace()
# abs_diff = (cdist - expand_express).abs().max()

# print(f"Max difference is {abs_diff}")




