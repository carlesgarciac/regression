from network.unet2D import UNet
from network.deeplabv3 import createDeepLabv3
from network.transUNet import transUNet
from network.cenet import CE_Net_
from network.CNN_LSTM import CNN_LSTM
from network.CNN_transformer import CNN_trans
from network.UNet_retrain import UNet_retrain
import torch
import copy

def network_manager(network_name, device):

    if network_name == 'UNet':
        model = UNet()

    elif network_name == 'Deeplabv3':
        model = createDeepLabv3()

    elif network_name == 'transUNet':
        model = transUNet()

    elif network_name == 'cenet':
        model = CE_Net_()

    elif network_name == 'CNN_LSTM':
        model = CNN_LSTM()

    elif network_name == 'CNN_retrain':
        pre_model = UNet()
        pre_model.load_state_dict(torch.load('/home/carlesgc/Projects/regression/models/MNMS_SA/UNet_final.pt'))
        # for params in pre_model.encoder1.parameters():
        #     print(params)

        model = UNet_retrain(pre_model)

        model.encoder1 = copy.deepcopy(pre_model.encoder1)
        model.encoder2 = copy.deepcopy(pre_model.encoder2)
        model.encoder3 = copy.deepcopy(pre_model.encoder3)
        model.encoder4 = copy.deepcopy(pre_model.encoder4)
        model.bottleneck = copy.deepcopy(pre_model.bottleneck)

        model.load_state_dict(torch.load('/home/carlesgc/Projects/regression/models/MNMS_SA/UNet_final.pt'), strict=False)

        for params in model.encoder1.parameters():
            params.requires_grad = False
        for params in model.encoder2.parameters():
            params.requires_grad = False
        for params in model.encoder3.parameters():
            params.requires_grad = False
        for params in model.encoder4.parameters():
            params.requires_grad = False
        for params in model.bottleneck.parameters():
            params.requires_grad = False

    elif network_name == 'CNN_trans':
        pre_model = UNet()
        pre_model.load_state_dict(torch.load('/home/carlesgc/Projects/regression/models/MNMS_SA/UNet_final.pt'))
        # for params in pre_model.encoder1.parameters():
        #     print(params)

        model = CNN_trans(pre_model)

        model.encoder1 = copy.deepcopy(pre_model.encoder1)
        model.encoder2 = copy.deepcopy(pre_model.encoder2)
        model.encoder3 = copy.deepcopy(pre_model.encoder3)
        model.encoder4 = copy.deepcopy(pre_model.encoder4)
        model.bottleneck = copy.deepcopy(pre_model.bottleneck)

        model.load_state_dict(torch.load('/home/carlesgc/Projects/regression/models/MNMS_SA/UNet_final.pt'), strict=False)

        for params in model.encoder1.parameters():
            params.requires_grad = False
        for params in model.encoder2.parameters():
            params.requires_grad = False
        for params in model.encoder3.parameters():
            params.requires_grad = False
        for params in model.encoder4.parameters():
            params.requires_grad = False
        for params in model.bottleneck.parameters():
            params.requires_grad = False

    model.to(device)
    return model