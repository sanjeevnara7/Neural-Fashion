from pureT_Fashion import PureT_Fashion
from xe_loss import LabelSmoothing
import torch
import numpy as np

#accepts predicted logits, target_seq
xe_loss_criterion = LabelSmoothing().cuda()

#accepts feature map from swin, input_seq, and gv_feat
B = 64  #batch_size
gv_feat = np.zeros((B, 1))
gv_feat = torch.cat([torch.from_numpy(b) for b in gv_feat], 0)

model = PureT_Fashion()
model = model.cuda()
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Trainable parameters: ", pytorch_total_params)

input_seq = torch.tensor([[1,2,3,4,5], [2,3,4,5,6]])

target_seq = torch.roll(input_seq, shifts = -1, dims = -1)

print(input_seq)
print(target_seq)