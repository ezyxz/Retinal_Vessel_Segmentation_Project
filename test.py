import os

import torch

from config_CHASE import model_dir

model = torch.load("Model_save/Fpn_unet_model_trained.pkl")
torch.save(model.state_dict(), os.path.join(model_dir, '%d.pth' % (130)))