from pureT_Fashion import PureT_Fashion
from xe_loss import LabelSmoothing
import torch
import numpy as np
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

#accepts predicted logits, target_seq
xe_loss_criterion = LabelSmoothing().cuda()

#accepts feature map from swin, input_seq, and gv_feat
B = 1  #batch_size
att_feat = torch.zeros((B, 64, 768)).cuda()
input_seq = torch.ones((B, 20), dtype = torch.int64).cuda()
gv_feat = np.zeros((B, 1))
gv_feat = torch.cat([torch.from_numpy(b) for b in gv_feat], 0).cuda()

model = PureT_Fashion()
model = model.cuda()

print(input_seq)
# softmax probabilities predicted by the model
output = model(att_feat, input_seq, gv_feat)
print("Model output during training: ", output.shape)

#computing the loss
loss = xe_loss_criterion(output, input_seq)
print("Training loss: ", loss)

#predictions during validation/testing
seq, _ = model.decode(att_feat)
print("Model validation prediction: ", seq)

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Number of parameters in PureT: ", pytorch_total_params)