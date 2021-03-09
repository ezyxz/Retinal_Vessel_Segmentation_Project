import re
import time
import torch
import tqdm
from config_DRIVE import * #DRIVE/CHASE
from networks import *
from dataset import *
import torchvision.transforms as standard_transforms
import transform as tr
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torch.autograd import Variable
import matplotlib.pyplot as plt
import cv2

from pred_big import test_my


def main():
    print("------------------------------")
    print("START")
    print("------------------------------")
    composed_transforms_tr = standard_transforms.Compose([
        tr.RandomHorizontalFlip(),
        tr.ScaleNRotate(rots=(-15, 15), scales=(.75, 1.5)),
        # tr.RandomResizedCrop(img_size),
        tr.FixedResize(img_size),
        tr.Normalize(mean=mean, std=std),
        tr.ToTensor()
    ])  # data pocessing and data augumentation

    voc_train_dataset = VOCSegmentation(base_dir=data_dir, split='train', transform=composed_transforms_tr)  # get data
    #return {'image': _img, 'gt': _target}
    print("Data loaded...")
    print("Dataset:{}".format(dataset))
    print("------------------------------")
    voc_train_loader = DataLoader(voc_train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    iter_dataset = iter(voc_train_loader)
    train = next(iter_dataset)

    print("Input size {}".format(train['image'].shape))
    print("Output size {}".format(train['gt'].shape))

    print("Model start training...")
    print("------------------------------")
    print("Model info:")
    print("If use CUDA : {}".format(use_gpu))
    print('Initial  learning rate {} | batch size {} | epoch num {}'.format(0.0001, batch_size, epoches))
    print("------------------------------")

    model = fpn_unet(input_bands=input_bands, n_classes=num_class)
    model_id = 0
    # load model
    if find_new_file(model_dir) is not None:
        model.load_state_dict(torch.load(find_new_file(model_dir)))
        # model.load_state_dict(torch.load('./pth/best2.pth'))
        print('load the model %s' % find_new_file(model_dir))
        model_id = re.findall(r'\d+', find_new_file(model_dir))
        model_id = int(model_id[0])
    print('Current model ID {}'.format(model_id))
    model.cuda()
    criterion = torch.nn.CrossEntropyLoss()   #define loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  #define optimizer
    model.cuda()
    model.train()
    f = open('log.txt', 'w')

    for epoch in range(epoches):
        cur_log = ''
        running_loss = 0.0
        start = time.time()
        lr = adjust_learning_rate(base_lr, optimizer, epoch, model_id, power)
        print("Current learning rate : {}".format(lr))
        for i, batch_data in tqdm.tqdm(enumerate(voc_train_loader)):#get data
            images, labels = batch_data['image'], batch_data['gt']
            labels = labels.view(images.size()[0], img_size, img_size).long()

            i += images.size()[0]
            images = Variable(images).cuda()
            labels = Variable(labels).cuda()
            optimizer.zero_grad()
            outputs = model(images)
            losses = criterion(outputs, labels)  # calculate loss
            losses.backward()
            optimizer.step()
            running_loss += losses

        print("Epoch [%d] all Loss: %.4f" % (epoch + 1 + model_id, running_loss / i))
        cur_log+='epoch:{}, '.format(str(epoch))+'learning_rate:{}'.format(str(lr))+', '+'train_loss:{}'.format(str(running_loss.item() / i))+', '
        torch.save(model.state_dict(), os.path.join(model_dir, '%d.pth' % (model_id + epoch + 1)))
        print("Model Saved")
        # iou, acc, recall, precision = test_my(input_bands, model_name, model_dir, img_size, num_class)
        # cur_log += 'iou:{}'.format(str(iou)) + ', ' + 'acc:{}'.format(str(acc))+'\n' + ', ' + 'recall:{}'.format(str(recall))+'\n' + ', ' + 'precision:{}'.format(str(precision))
        end = time.time()
        time_cha = end - start
        left_steps = epoches - epoch - model_id
        print('the left time is %d hours, and %d minutes' % (int(left_steps * time_cha) / 3600,
                                                             (int(left_steps * time_cha) % 3600) / 60))

        print(cur_log)
        f.writelines(str(cur_log))


def find_new_file(dir):
    file_lists = os.listdir(dir)
    file_lists.sort(key=lambda fn: os.path.getmtime(dir + fn)
                    if not os.path.isdir(dir + fn) else 0)
    if len(file_lists) != 0:
        file = os.path.join(dir, file_lists[-1])
        return file
    else:
        return None


def adjust_learning_rate(base_lr, optimizer, epoch, model_id, power):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # lr = base_lr * ((1-float(epoch+model_id)/num_epochs)**power)
    lr = base_lr * (power ** ((epoch+model_id) // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def predcite_and_cal():
    iou, acc, recall, precision = test_my(input_bands, model_name, model_dir, img_size, num_class)
    cur_log = 'iou:{}'.format(str(iou)) + ', ' + 'acc:{}'.format(str(acc))+'\n' + ', ' + 'recall:{}'.format(str(recall))+'\n' + ', ' + 'precision:{}'.format(str(precision))
    print(cur_log)

if __name__ == '__main__':
    main()
    #predcite_and_cal()
    print("exit..")