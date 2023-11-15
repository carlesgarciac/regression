from unicodedata import bidirectional
import torch
from torch import nn

from collections import OrderedDict

class CNN_LSTM(nn.Module):
    def __init__(self, in_channels=1, init_features=64):
        super(CNN_LSTM, self).__init__()

        features = init_features

        self.encoder1 = CNN_LSTM._block(in_channels, features, kernel_size=7, stride=2, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.encoder2 = CNN_LSTM._block(features, features*2, kernel_size=5, stride=2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.encoder3 = CNN_LSTM._block(features*2, features*4 , kernel_size=3, stride=1, name="enc3")
        self.encoder4 = CNN_LSTM._block(features*4, features*8, kernel_size=3, stride=1, name="enc4")
        self.encoder5 = CNN_LSTM._block(features*8, features*16, kernel_size=3, stride=1, name="enc5")
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.fc1 = nn.Linear(25600,1024)

        self.relu = nn.ReLU()

        self.lstm = nn.LSTM(input_size=1024, hidden_size=1, num_layers=2, batch_first=True, bidirectional=False)

        self.fc2 = nn.Linear(1024, 1)

    def forward(self, x):
        hidden = (torch.zeros(2, 1, 1).cuda(),
            torch.zeros(2, 1, 1).cuda())
        out = []
        # Inside for to lstm
        # print(x.shape[1])
        # print(x.shape)
        for i in range(x.shape[1]):
            enc1 = self.encoder1(x[:,i,:,:,8].view(1,1,224,224))
            # print(enc1.size())
            enc2 = self.encoder2(self.pool1(enc1))
            # print(enc2.size())
            enc3 = self.encoder3(self.pool2(enc2))
            # print(enc3.size())
            enc4 = self.encoder4(enc3)
            # print(enc4.size())
            enc5 = self.encoder5(enc4)
            # print(enc5.size())
            enc6 = self.pool3(enc5)
            # print(enc6.size())
            viewed = enc6.view(1,-1)
            # print(viewed.size())
            fc1 = self.fc1(viewed)
            # print(fc1.size())
            fc1_a = self.relu(fc1)
            # print(fc1_a.size())
            # print(fc1_a.view(-1,1,1).size())
            lstm_out, hidden = self.lstm(fc1_a.view(1, 1, -1), hidden)
            out.append(self.relu(lstm_out))
            # print(lstm)
            # if i != 0:
            #     out = torch.cat((out,lstm), 0, requires_grad=True)
            # else:
            #     out = lstm

        
        # out = torch.cat(out, 0, requires_grad=True)
        out = torch.stack(out, 0)
        # print(out)

        # out_ED = out.argmax(axis=0)
        # out_ES = out.argmin(axis=0)

        # print(lstm.size())
        
        # outside for
        # fc2 = self.fc2(lstm)
        # out = self.relu(fc2)
        # out = torch.Tensor(out)
        # print(len(out))
        # print(out)
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
