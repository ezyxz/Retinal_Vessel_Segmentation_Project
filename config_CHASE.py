import torch
batch_size = 6
use_gpu = torch.cuda.is_available()

img_size = 256
epoches = 30
base_lr = 0.0001
weight_decay = 2e-5
momentum = 0.9
power = 0.99

num_class = 2  # some parameters
model_name = 'Fpn_unet'
input_bands = 3
if input_bands==4:
    mean = (0.315, 0.319, 0.470, 0.357)
    std = (0.144, 0.151, 0.211, 0.195)
else:
    mean = (0.315, 0.319, 0.470)
    std = (0.144, 0.151, 0.211)
# dataset = 'CHASE'
dataset = 'CHASE'
data_dir='./data_slice_{}'.format(dataset)
val_path = './{}/test/images/'.format(dataset)
output = './result_{}/'.format(model_name)
output_gray = './result_gray_{}/'.format(model_name)
val_gt = './{}/test/1st_manual/'.format(dataset)
model_dir = './pth_{}/'.format(model_name)