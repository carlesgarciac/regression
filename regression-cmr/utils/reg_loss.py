import torch
from torch import Tensor
from kornia.utils import one_hot
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt

def reg_loss(prediction, ED, ES, device):
    # print(prediction)
    prediction_toSyn = prediction.squeeze().detach().cpu().numpy()
    y_k = synthetic_label(prediction_toSyn, ED.numpy(), ES.numpy())
    
    # print(prediction.squeeze())
    # print(y_k)
    # print(ED)
    # print(ES)
    # print('-----')
    mse_loss = F.mse_loss(prediction.squeeze(), y_k.to(device))
    temp_loss = ltemp(y_k, prediction_toSyn)

    loss = mse_loss + temp_loss
    
    return loss


def synthetic_label(prediction, ED, ES):
    y_k = []
    for k in range(len(prediction)):
        if (int(ED) < k) and (k <= int(ES)):
            y_k.append((abs((k-ES)/(ES-ED)))**3)
            # print(1)
        else:
            y_k.append((abs((k-ES)/(ES-ED)))**(1/3))
    # print(y_k)
    # plt.plot(y_k)
    # plt.savefig('y_k.png')
    return torch.from_numpy(np.array(y_k, dtype= "float32"))

def ltemp(y_k, prediction):
    Linc = linc(y_k, prediction)
    Ldec = ldec(y_k, prediction)
    ltemp = (Linc+Ldec)/2
    # print(ltemp)
    return torch.from_numpy(np.array(ltemp, dtype= "float32"))


def linc(y_k, prediction):
    Linc = 0
    for k in range(len(prediction)-1):
        if y_k[k+1] > y_k[k]:
            Linc = Linc + max(0,prediction[k]-prediction[k+1])
            # print('linc')
    return Linc/len(prediction)

def ldec(y_k, prediction):
    Ldec = 0
    for k in range(len(prediction)-1):
        if y_k[k+1] < y_k[k]:
            Ldec = Ldec + max(0,prediction[k+1]-prediction[k])
            # print('ldec')
    return Ldec/len(prediction)
