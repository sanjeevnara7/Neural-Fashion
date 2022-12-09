from pureT_Fashion import PureT_Fashion
from xe_loss import LabelSmoothing
import torch
import numpy as np

#accepts predicted logits, target_seq
xe_loss_criterion = LabelSmoothing().cuda()

#accepts feature map from swin, input_seq, and gv_feat
B = 1  #batch_size
gv_feat = np.zeros((B, 1))
gv_feat = torch.cat([torch.from_numpy(b) for b in gv_feat], 0).cuda()

model = PureT_Fashion()
model = model.cuda()
att_feat = torch.zeros((1, 64, 768)).cuda()
inp_seq = torch.ones((1, 20), dtype = torch.int32).cuda()
print(inp_seq)
output = model(att_feat, inp_seq, gv_feat)
seq, _ = model.decode(att_feat)
print(seq)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Trainable parameters: ", pytorch_total_params)

# input_seq = torch.tensor([[1,2,3,4,5], [2,3,4,5,6]])

# target_seq = torch.roll(input_seq, shifts = -1, dims = -1)

# print(input_seq)
# print(target_seq)