import os
import torch 
import numpy as np
from PIL import Image, ImageFilter
from torchvision.transforms import ToTensor
import torch.nn.functional as F

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

def apply_disparity(self, img:torch.Tensor, disp:torch.Tensor):
        batch_size, _, height, width = img.size()

        # Original coordinates of pixels
        x_base = torch.linspace(0, 1, width).repeat(batch_size,
                    height, 1).type_as(img)
        y_base = torch.linspace(0, 1, height).repeat(batch_size,
                    width, 1).transpose(1, 2).type_as(img)

        # Apply shift in X direction
        x_shifts = disp[:, 0, :, :]  # Disparity is passed in NCHW format with 1 channel
        flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)
        # In grid_sample coordinates are assumed to be between -1 and 1
        output = F.grid_sample(img, 2*flow_field - 1, mode='bilinear',
                               padding_mode='zeros')

        return output