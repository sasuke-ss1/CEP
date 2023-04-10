import os
from torch.utils.data import Dataset
from PIL import Image, ImageOps
import random


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