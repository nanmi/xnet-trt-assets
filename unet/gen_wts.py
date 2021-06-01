import torch
from torch import nn
import torchvision
import os
import struct
from torchsummary import summary

def main():
    print('cuda device count: ', torch.cuda.device_count())
    net = torch.load('../Pytorch-UNet/models/unet_carvana_scale1_epoch5.pth')

    f = open("unet.wts", 'w')
    f.write("{}\n".format(len(list(net))))
    for k,v in net.items():
        print('key: ', k)
        print('value: ', v.shape)
        vr = v.reshape(-1).cpu().numpy()
        f.write("{} {}".format(k, len(vr)))
        for vv in vr:
            f.write(" ")
            f.write(struct.pack(">f", float(vv)).hex())
        f.write("\n")

if __name__ == '__main__':
    main()
