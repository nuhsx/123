import torch
import Nnet
model=Nnet.nnet(6)
reload_states = torch.load("state.pt")
model=model.load_state_dict(reload_states)

model.load_state_dict(checkpoint['net'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']



