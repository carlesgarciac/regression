from unicodedata import bidirectional
import torch
from torch import nn

from collections import OrderedDict

class UNet_retrain(nn.Module):
    def __init__(self, pre_model, in_channels=1, init_features=32):
        super(UNet_retrain, self).__init__()

        features = init_features
        
        self.encoder1 = pre_model.encoder1
        self.pool1 = pre_model.pool1
        self.encoder2 = pre_model.encoder2
        self.pool2 = pre_model.pool2
        self.encoder3 = pre_model.encoder3
        self.pool3 = pre_model.pool3
        self.encoder4 = pre_model.encoder4
        self.pool4 = pre_model.pool4
        self.bottleneck = pre_model.bottleneck
        
        # self.CNN = nn.Sequential(
        #     self.encoder1,
        #     self.pool1,
        #     self.encoder2,
        #     self.pool2,
        #     self.encoder3,
        #     self.pool3,
        #     self.encoder4,
        #     self.pool4,
        #     self.botteleck
        #     )

        self.fc1 = nn.Linear(131072,512)

        self.relu = nn.ReLU()

        self.lstm = nn.LSTM(input_size=512, hidden_size=512, num_layers=2, bidirectional=True)

        self.fc2 = nn.Linear(1024, 1)

    def forward(self, x):
        # hidden = (torch.zeros(2, 1, 1).cuda(),
        #     torch.zeros(2, 1, 1).cuda())
        cnn = []
        # Inside for to lstm
        # print(x.shape[1])
        # print(x.shape)
        
        enc1 = self.encoder1(x[:,:,:,:,8].view(x.shape[1],1,256,256))
        enc2 = self.encoder2(self.pool1(enc1))
        # print(enc2.size())
        enc3 = self.encoder3(self.pool2(enc2))
        # print(enc3.size())
        enc4 = self.encoder4(self.pool3(enc3))
        # print(enc4.size())
        enc5 = self.bottleneck(self.pool4(enc4))
        # print(enc5.size())
        viewed = enc5.view(-1, 131072)
        # print(viewed.size())
        fc1 = self.fc1(viewed)
        # print(fc1.size())
        fc1_a = self.relu(fc1)

        # for i in range(x.shape[1]):
        #     enc1 = self.encoder1(x[:,i,:,:,8].view(1,1,256,256))
        #     # print(enc1.size())
        #     enc2 = self.encoder2(self.pool1(enc1))
        #     # print(enc2.size())
        #     enc3 = self.encoder3(self.pool2(enc2))
        #     # print(enc3.size())
        #     enc4 = self.encoder4(self.pool3(enc3))
        #     # print(enc4.size())
        #     enc5 = self.bottleneck(self.pool4(enc4))
        #     # print(enc5.size())
        #     viewed = enc5.view(1,-1)
        #     # print(viewed.size())
        #     fc1 = self.fc1(viewed)
        #     # print(fc1.size())
        #     fc1_a = self.relu(fc1)
        #     print(fc1_a)
        #     print('------')
        #     # print(fc1_a.view(-1,1,1).size())
        #     cnn.append(fc1_a)
        
        # to_lstm = torch.stack(cnn, 0)
        # print(to_lstm)
        to_lstm = fc1_a.unsqueeze(1)
        # breakpoint()
        lstm_out, _ = self.lstm(to_lstm)
        # out.append(lstm_out)
            # print(lstm)
            # if i != 0:
            #     out = torch.cat((out,lstm), 0, requires_grad=True)
            # else:
            #     out = lstm
        linear2 = self.fc2(lstm_out)
        out = linear2.squeeze()
        # out = torch.cat(out, 0, requires_grad=True)
        # out = torch.stack(out, 0)
        # print(out)

        # out_ED = out.argmax(axis=0)
        # out_ES = out.argmin(axis=0)

        # print(lstm.size())
        
        # outside for
        # fc2 = self.fc2(lstm)
        # out = self.relu(fc2)
        # out = torch.Tensor(out)
        # print(len(out))
        # print(lstm_out)
        return out



    def _block(in_channels, features, name, kernel_size, stride, padding=1, bias=True):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding,
                            bias=bias,
                        ),
                    ),
                    # (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),   
                ]
            )
        )
