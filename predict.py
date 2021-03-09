import cv2
import torch
import time
import torch
import tqdm
from config_CHASE import *
import torchvision.transforms as transforms
from networks import *
from dataset import *
import torchvision.transforms as standard_transforms
import transform as tr
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
image_dir = './CHASE/test/images/01_test.tiff'
def img_transforms(img):
    img = np.array(img).astype(np.float32)
    sample = {'image': img}
    # img, label = random_crop(img, label, crop_size)
    transform = transforms.Compose([
        # tr.FixedResize(img_size),
        tr.Normalize(mean=mean, std=std),
        tr.ToTensor()])
    sample = transform(sample)
    return sample['image']

def predict_ndarray(net, im): # 预测结果
    use_gpu = False
    with torch.no_grad():
        if use_gpu:
            im = im.unsqueeze(0).cuda()
        else:
            im = im.unsqueeze(0)
        output = net(im)
        pred = output.max(1)[1].squeeze().cpu().data.numpy()
        # pred_ = label_mapping(pred)
    return pred

def load_img(dir):
    img = np.asarray(Image.open(dir).convert('RGB')).astype(np.float32)
    return img
def predict():
    img = load_img(image_dir)
    plt.imshow(img.astype(np.int))
    plt.show()

    train = img_transforms(img)
    print(train.size())

    model = torch.load('Model_save\Fpn_unet_model_trained.pkl')
    model.cpu()
    img_pred = predict_ndarray(model, train)
    np.save("Test_predicted/test_pred1.npy", img_pred)
    print("complete...")

def img_plt():
    predicted = np.load("Test_predicted/test_pred1.npy")
    print(predicted.shape)
    # cv2.imshow('cv2_img', predicted)
    # cv2.waitKey(0)
    plt.imshow(predicted,cmap ='gray')
    plt.show()

def manmual_plt():
    manmual1 = 'CHASE/test/1st_manual/01_manual1.gif'
    img = np.asarray(Image.open(manmual1).convert('RGB'))
    plt.imshow(img,cmap ='gray')
    plt.show()
if __name__ == '__main__':
    # predict()
    # img_plt()
    manmual_plt()
    print("exit....")