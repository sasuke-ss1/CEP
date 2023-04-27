import torch
import torch.nn as nn


class Test(nn.Module):
    def __init__(self, out=64):
        super().__init__()
        sizes= [3, out, out, out, out, 3] 
        layers =[]
        for i in range(4):
            layers.append(RefConvInsact(sizes[i], sizes[i+1], 3, 1, 0))
        layers.append(RefConvInsact(sizes[4], sizes[5], 1, 1, 0, False,True, False))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    
class Dest(nn.Module):
    def __init__(self, out=64):
        super().__init__()
        sizes= [3, out, out, out, out, 1] 
        layers =[]
        for i in range(4):
            layers.append(RefConvInsact(sizes[i], sizes[i+1], 3, 1, 0))
        layers.append(RefConvInsact(sizes[4], sizes[5], 1, 1, 0, False,True, False))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class Jest(nn.Module):
    def __init__(self, out=64):
        super().__init__()
        sizes= [3, out, out, out, out, 3] 
        layers =[]
        for i in range(4):
            layers.append(RefConvInsact(sizes[i], sizes[i+1], 3, 1, 0))
        layers.append(RefConvInsact(sizes[4], sizes[5], 1, 1, 0, False,True, False))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class RefConvInsact(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, useNorm=True, useSigmoid=False, useReflection=True):
        super().__init__()

        self.useNorm = useNorm
        self.useReflection = useReflection 
            
        self.reflection = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding)
        self.norm = nn.InstanceNorm2d(out_channel)
        self.act = nn.Sigmoid() if useSigmoid else nn.ReLU()

    def forward(self, x):
        
        if self.useReflection:
            x = self.reflection(x)
        x = self.conv(x)
        if self.useNorm:
            x = self.norm(x)
        out = self.act(x)

        return out

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.Tx = Test()
        self.J = Jest()
        self.D = Dest()

    def forward(self, x):
        t = self.Tx(x)
        j = self.J(x)
        d = self.D(x)

        return j, t, d