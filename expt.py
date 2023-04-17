import cv2
import matplotlib.pyplot as plt
from utils import np2pil, torch2np
import numpy as np
from Model import Net
import torch


#J_path = "results/J/889.jpg"
#T_path = "results/T/889.jpg"
#B_path = "results/A/889.jpg"
#
#J = cv2.imread(J_path)/255
#T = cv2.imread(T_path)/255
#B = cv2.imread(B_path)/255
#
#J = np.transpose(J, (2, 0, 1))
#T = np.transpose(T, (2, 0, 1))
#B = np.transpose(B, (2, 0, 1))
#
#I = J*T + B*(1-T)
#img = np2pil(I).save("./123.png")
#
##plt.imshow(I)
##plt.savefig("./123.png")



#model = Net()
#model.load_state_dict(torch.load("weights/epoch_200.pth"))
#
#
#img = np.transpose(cv2.imread("/home/intern/ss_sasuke/CEP/UIE-dataset/UIEBD/train/image/237.jpg"), (2, 0, 1))
#img = torch.from_numpy(img)
#j, t = model(img)
#
#j, t = np2pil(torch2np(j)), np2pil(torch2np(t))
#
#j.save("./123.png")
#t.save("./123.png")




