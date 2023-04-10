import os
import torch 
import numpy as np
from PIL import Image, ImageFilter
from torchvision.transforms import ToTensor

def np2pil(npImage):
    npImage = np.clip(npImage*255, 0, 255).astype(np.uint8)
    
    assert npImage.shape[0] in [1, 3]

    if npImage.shape[0] == 3:
        npImage = npImage.transpose(1, 2, 0)
    else:
        npImage = npImage[0]
    
    return Image.fromarray(npImage)

def save_img(name, npImage, path):
    if not os.path.exists(path):
        os.mkdir(path)

    pilImage = np2pil(npImage)
    pilImage.save(path+ f"{name}")

def np2torch(npImage):
    return torch.from_numpy(npImage)[None, ...]

def torch2np(torchImage):
    return torchImage.detach().cpu().numpy()[0]

def get_A(x):
    npX = np.clip(torch2np(x), 0, 1)
    pilX = np2pil(npX)

    h, w = pilX.size
    window = 0.5*(h+w)

    A = pilX.filter(ImageFilter.GaussianBlur(window))
    A = ToTensor()(A)
    
    return A.unsqueeze(0)