import os
from torch.utils.data import Dataset
from PIL import Image, ImageOps
import random
import torchvision.transforms.functional as TF
from torchvision.transforms import RandomCrop
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

def get_patch(img_in, img_tar, patch_size, scale=1, ix=-1, iy=-1):
    (ih, iw) = img_in.size

    patch_mult = scale
    tp = patch_mult * patch_size
    ip = tp // scale

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    (tx, ty) = (scale * ix, scale * iy)

    img_in = img_in.crop((iy, ix, iy + ip, ix + ip))
    img_tar = img_tar.crop((ty, tx, ty + tp, tx + tp))

    info_patch = {
        'ix': ix, 'iy': iy, 'ip': ip, 'tx': tx, 'ty': ty, 'tp': tp}

    return img_in, img_tar, info_patch




class Data(Dataset):
    def __init__(self, data_dir, label_dir, patch_size, transform=None):
        super().__init__()
        data_filenames = [os.path.join(data_dir, x) for x in os.listdir(data_dir)]
        data_filenames.sort()
        self.data_filenames = data_filenames
        label_filenames = [os.path.join(label_dir, x) for x in os.listdir(label_dir)]
        label_filenames.sort()
        self.label_filenames = label_filenames
        self.patch_size = patch_size
        self.transform = transform
        

    def __getitem__(self, index):
        target = Image.open(self.label_filenames[index]).convert('RGB')
        input = Image.open(self.data_filenames[index]).convert('RGB')
        _, file = os.path.split(self.label_filenames[index])
        input, target, _ = get_patch(input, target, self.patch_size)

        if self.transform:
            input = self.transform(input)
            target = self.transform(target)

        return input, target, file

    def __len__(self):
        return len(self.label_filenames)

class Val(Dataset):
    def __init__(self, data_dir, label_dir, transform=None):
        super().__init__()
        data_filenames = [os.path.join(data_dir, x) for x in os.listdir(data_dir)]
        data_filenames.sort()
        self.data_filenames = data_filenames
        label_filenames = [os.path.join(label_dir, x) for x in os.listdir(label_dir)]
        label_filenames.sort()
        self.label_filenames = label_filenames
        self.transform = transform

    def __getitem__(self, index):
        target = Image.open(self.label_filenames[index]).convert("RGB")
        input = Image.open(self.data_filenames[index]).convert("RGB")
        _, file = os.path.split(self.label_filenames[index])

        if self.transform:
            input = self.transform(input)
            target = self.transform(target)

        return input, target, file

    def __len__(self):
        return len(self.label_filenames)
    
class StereoData(Dataset):
    def __init__(self, root, patch_size=None, val=False, transform=None):
        super().__init__()
        
        root = os.path.join(root, "val" if val else "train")
        self.val = val

        self.left = os.path.join(root, "left")
        self.right = os.path.join(root, "right")
        self.leftImagePath = list(map(lambda x: os.path.join(self.left, x), sorted(os.listdir(self.left))))
        self.rightImagePath = list(map(lambda x: os.path.join(self.right, x), sorted(os.listdir(self.right))))

        self.patch_size = patch_size
        self.transform = transform

    def __getitem__(self, idx):
        imgL = Image.open(self.leftImagePath[idx])
        imgR = Image.open(self.rightImagePath[idx])

        if not self.val:
            imgL, imgR, _ =  get_patch(imgL, imgR, self.patch_size)

        if self.transform:
            imgL, imgR = self.transform(imgL), self.transform(imgR)

        if self.val:
            return imgL, imgR, self.leftImagePath[idx].split("/")[-1]

        return imgL, imgR

    def __len__(self):
        return len(self.leftImagePath)
    
if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor()])
    imgL, imgR = StereoData("DROPUWStereo_HIMB_Data", 128, transform=transform)[0]
    print(imgL.shape, imgR.shape)
