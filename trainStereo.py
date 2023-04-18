import os
from torch.utils.data import DataLoader
from Model import Net
import yaml
import torch.nn as nn
from argparse import ArgumentParser
from torch.optim import Adam
from tqdm import tqdm
from losses import GrayLoss
from utils import *
from torch.optim.lr_scheduler import MultiStepLR
from dataset import StereoData
from torchvision import transforms
import torchvision.transforms.functional as TF
import sys

parser = ArgumentParser()
parser.add_argument("--configPath", "-p", default="./config.yml", type=str)

args = parser.parse_args()

with open(args.configPath, "r") as f:
    config = yaml.safe_load(f)


gray_loss = GrayLoss()
mse_loss = nn.MSELoss()

model = Net().cuda()
optimizer = Adam(model.parameters(), lr=config["lr"])

ckpt = [i for i in range(1, config["Epochs"] + 1) if i%config["decay"] == 0]
scheduler = MultiStepLR(optimizer, ckpt, config["gamma"])

transform = transforms.Compose([transforms.ToTensor()])


trainData = StereoData(config["trainDir"], config["patchSize"], transform=transform)
trainLoader = DataLoader(trainData, batch_size=config["batchSize"], shuffle=True)
valData = StereoData(config["valDir"] , val=True,transform=transform)
valLoader = DataLoader(valData)

def train():    
    for epoch in range(1, config["Epochs"] + 1):
        model.train()
        loss = 0
        loop_obj = tqdm(trainLoader)
        for batch in loop_obj:
            loop_obj.set_description(f"Epoch: {epoch}")
            imgL, imgR = batch[0].cuda(), batch[1].cuda()
            j, t = model(imgL)
            a = get_A(imgL).cuda()

            #rgb = [torch.exp(-d*betas[..., i].view(-1, 1, 1, 1)) for i in range(betas.shape[1])]
            #t = torch.cat(rgb, dim=1)
            
            IRec = j * t + (1 - t) * a
            loss_1 = mse_loss(IRec, imgL)
            d = TF.rgb_to_grayscale(t)
            # Stereo Matching loss
            imgRrecon = apply_disparity(imgL, d)
            loss4 = mse_loss(imgRrecon, imgR)

            lam = np.random.beta(1, 1)
            imgL_mix = lam * imgL + (1 - lam) * j

            jMix, tMix = model(imgL_mix)
            loss_2 = mse_loss(jMix, j.detach())

            loss_3 = gray_loss(j)

            total_loss = 1 * loss_1 + 1 * loss_2 + 0.01 * loss_3 + 1 * loss4

            optimizer.zero_grad()
            total_loss.backward()
            loss += total_loss.item()
            optimizer.step()
            loop_obj.set_postfix_str(f"loss: {loss:0.3f}")

        weightsPath = config["saveFolder"]+f"epoch_{epoch}.pth"  
        if not os.path.exists(config["saveFolder"]):
            os.mkdir(config["saveFolder"])
        torch.save(model.state_dict(), weightsPath)

        if epoch % config["testFreq"] == 0:
            print("Starting Validation")
            torch.set_grad_enabled(False)
            model.eval()
            for batch in tqdm(valLoader):
                imgL, imgR, name = batch[0].cuda(), batch[1], batch[2]
                with torch.no_grad():
                    j_out, t_out = model(imgL)
                    a_out = get_A(imgL).cuda()

                    #rgb_out = [torch.exp(-d_out*betas[..., i].view(-1, 1, 1, 1)) for i in range(betas.shape[1])]
                    #t_out = torch.cat(rgb_out, dim=1)
                    d_out = TF.rgb_to_grayscale(t)
                    imgRrecon = apply_disparity(imgL, d_out)

                    if not os.path.exists(config["outputFolder"]):
                        os.mkdir(config["outputFolder"])
                        os.mkdir(config["outputFolder"] + 'J/')
                        os.mkdir(config["outputFolder"] + 'A/')
                        os.mkdir(config["outputFolder"] + 'T/')
                        #os.mkdir(config["outputFolder"] + "D/")
                        os.mkdir(config["outputFolder"] + "Ir/")
                    j_out_np = np.clip(torch2np(j_out), 0, 1)
                    t_out_np = np.clip(torch2np(t_out), 0, 1)
                    #d_out_np = np.clip(torch2np(d_out), 0, 1)
                    a_out_np = np.clip(torch2np(a_out), 0, 1)
                    ir_out_np = np.clip(torch2np(imgRrecon), 0, 1)
                    save_img(name[0], j_out_np, config["outputFolder"] + 'J/')
                    save_img(name[0], t_out_np, config["outputFolder"] + 'T/')
                    save_img(name[0], a_out_np, config["outputFolder"] + 'A/')
                    #save_img(name[0], d_out_np, config["outputFolder"] + 'D/')
                    save_img(name[0], ir_out_np, config["outputFolder"] + 'Ir/')
                    
            torch.set_grad_enabled(True)


if __name__ == "__main__":
    train()