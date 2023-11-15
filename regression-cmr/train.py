import random
from tkinter import Variable

import time

from tqdm import tqdm

import torch
from torch import argmin, ne, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torchio as tio

from utils.reg_loss import reg_loss
from utils.logging import image_logger_val, image_logger_train
from utils.checker import dataloader_checker

from data.loader_manager import loader_manager

from network.network_manager import network_manager

import statistics

import neptune

## Params
seed = 42
random.seed(seed)
torch.manual_seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## Available architectures: UNet, Deeplabv3, transUNet, cenet
network_name = 'CNN_trans'
print(network_name)
## Available views: SA, LA 
data_view = 'SA'

model = network_manager(network_name, device)

epochs = int(500)
batch_size = int(1)
num_workers = int(8)
learning_rate = float(0.0001)
save_checkpoint = bool(True)
training_split_ratio = 0.8

## Datasets
train_loader, val_loader, test_loader = loader_manager(data_view, batch_size, num_workers, create_landmarks=False)

## Logging (neptune)
neptune.init(project_qualified_name="carlesgc/Regression")
neptune.create_experiment(network_name+'_'+data_view)

## Optimer, Loss, lr scheduler, etc
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# scheduler = ReduceLROnPlateau(optimizer)
loss_function = reg_loss

## Training
def acc_calculation(prediction, ed, es):
    pred_ed = prediction.argmax(axis=0) 
    pred_es = prediction.argmin(axis=0) 
    if abs(pred_ed - ed) > (len(prediction)/2):
        acc_ed = len(prediction) - pred_ed + ed + 1
    else:
        acc_ed = abs(pred_ed - ed)
    # acc_ed = abs(pred_ed - ed)
    acc_es = abs(pred_es - es)
    return acc_ed, acc_es

def prepare_batch(batch, device):
    inputs = batch['mri'][tio.DATA].to(device).squeeze(4)
    targets = batch['heart'][tio.DATA].to(device).squeeze(4)
    ED = batch['ED'].squeeze()
    ES = batch['ES'].squeeze()
    patient = batch['patient']
    
    if inputs.shape[1] > 40:
        print(inputs.shape)
        print(patient)
    return inputs, targets, ED, ES, patient

def train(model, device, train_loader, optimizer, loss_function, epoch):
    model.train()
    train_loss = []
    train_acc_ed = []
    train_acc_es = []
    with tqdm(train_loader, unit="batch") as tepoch:
        for batch_idx, batch in enumerate(tepoch): #enumerate(tqdm(train_loader, desc='Epoch '+str(epoch)+'/'+str(epochs))):
            tepoch.set_description('Epoch '+str(epoch)+'/'+str(epochs))

            
            data, target, ED, ES, patient = prepare_batch(batch, device)
            optimizer.zero_grad()

            # print(patient)
            # print(data.shape)

            if network_name == 'Deeplabv3' or network_name == 'cenet':
                data = data.expand(-1,3,-1,-1)
            
            output = model(data)

            if network_name == 'Deeplabv3':
                output = output['out']

            loss = loss_function(output, ED, ES, device)
            # print(loss)
            # print(type(loss))

            # loss.requires_grad=True
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

            acc_ed, acc_es = acc_calculation(output, ED, ES)
            train_acc_ed.append(acc_ed.item())
            train_acc_es.append(acc_es.item())

            if batch_idx % 25 == 0:
                image_logger_train(data, output, target)
                # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset),100. * batch_idx / len(train_loader), loss.item()))
            
            tepoch.set_postfix(loss=statistics.mean(train_loss), acc_ed = statistics.mean(train_acc_ed), acc_es=statistics.mean(train_acc_es))  #, accuracy=100. * accuracy)
            
    return statistics.mean(train_loss), statistics.mean(train_acc_ed), statistics.mean(train_acc_es)

## Eval
def eval(model, device, loss_function, val_loader, epoch):
    model.eval()
    val_loss = []
    val_acc_ed = []
    val_acc_es = []
    with torch.no_grad():
        with tqdm(val_loader, unit="batch") as vepoch:
            for batch_idx, batch in enumerate(vepoch):
                vepoch.set_description('Validation '+str(epoch)+'/'+str(epochs))

                data, target, ED, ES, patient = prepare_batch(batch, device)

                if network_name == 'Deeplabv3' or network_name == 'cenet':
                    data = data.expand(-1,3,-1,-1)

                output = model(data)

                if network_name == 'Deeplabv3':
                    output = output['out']

                val_loss.append(loss_function(output, ED, ES, device).item()) 

                acc_ed, acc_es = acc_calculation(output, ED, ES)
                val_acc_ed.append(acc_ed.item())
                val_acc_es.append(acc_es.item())

                if batch_idx % 10 == 0:
                    image_logger_val(data, output, target)
                
                vepoch.set_postfix(
                    loss=statistics.mean(val_loss),
                    val_acc_ed=statistics.mean(val_acc_ed), 
                    val_acc_es=statistics.mean(val_acc_es) 
                    )
                                    
    val_loss = statistics.mean(val_loss)
    val_acc_ed = statistics.mean(val_acc_ed)
    val_acc_es = statistics.mean(val_acc_es)

    # print('\nValidation set: Average loss: {:.4f}\n'.format(val_loss))
    # print('\nValidation set: Average dice score: {:.4f}\n'.format(dice_all))
    return val_loss, val_acc_ed, val_acc_es

# Test
def test(model, device, test_loader):
    model.eval()
    # val_loss = []
    test_acc_ed = []
    test_acc_es = []
    time_elapsed = []
    with torch.no_grad():
        with tqdm(test_loader, unit="batch") as vepoch:
            for batch_idx, batch in enumerate(vepoch):
                vepoch.set_description('Test')

                data, target, ED, ES, patient = prepare_batch(batch, device)

                if network_name == 'Deeplabv3' or network_name == 'cenet':
                    data = data.expand(-1,3,-1,-1)

                start = time.time()
                output = model(data)
                end = time.time()

                if network_name == 'Deeplabv3':
                    output = output['out']

                # val_loss.append(loss_function(output, ED, ES, device).item()) 
                time_elapsed.append(end - start)

                acc_ed, acc_es = acc_calculation(output, ED, ES)
                test_acc_ed.append(acc_ed.item())
                test_acc_es.append(acc_es.item())

                # if batch_idx % 10 == 0:
                #     image_logger_val(data, output, target)
                
                vepoch.set_postfix(
                    time=statistics.mean(time_elapsed),
                    test_acc_ed=statistics.mean(test_acc_ed), 
                    test_acc_es=statistics.mean(test_acc_es) 
                    )
                                    
    # val_loss = statistics.mean(test_loss)
    test_time = statistics.mean(time_elapsed)
    test_acc_ed = statistics.mean(test_acc_ed)
    test_acc_es = statistics.mean(test_acc_es)

    # print('\nValidation set: Average loss: {:.4f}\n'.format(val_loss))
    # print('\nValidation set: Average dice score: {:.4f}\n'.format(dice_all))
    return test_time, test_acc_ed, test_acc_es

for epoch in range(1, epochs):
    train_loss, train_acc_ed, train_acc_es = train(model, device, train_loader, optimizer, loss_function, epoch)
    val_loss, val_acc_ed, val_acc_es = eval(model, device, loss_function, val_loader, epoch)
    # scheduler.step(val_loss)

    if epoch %  5 == 0  and save_checkpoint:
        torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': train_loss,    
        }, '/home/carlesgc/Projects/regression/models/'+'Regression'+'/'+network_name+'_'+format(epoch,"04")+'.pt')

    # dataloader_checker(val_loader)

    neptune.log_metric('train_loss', train_loss)
    neptune.log_metric('train_acc_ed', train_acc_ed)
    neptune.log_metric('train_acc_es', train_acc_es)
    neptune.log_metric('val_loss', val_loss)
    neptune.log_metric('val_acc_ed', val_acc_ed)
    neptune.log_metric('val_acc_es', val_acc_es)

torch.save(model.state_dict(), '/home/carlesgc/Projects/regression/models/'+'Regression'+'/'+network_name+'_final.pt')


## Testing
# lstm
# model.load_state_dict(torch.load('/home/carlesgc/Projects/regression/models/Regression/CNN_retrain_final.pt'))
# Transformer
# model.load_state_dict(torch.load('/home/carlesgc/Projects/regression/models/Regression/CNN_trans_final.pt'))

# test_time, test_acc_ed, test_acc_es = test(model, device, test_loader)

# print('Test time:'+str(test_time))
# print('Test Acc ED:'+str(test_acc_ed))
# print('Test ACC ES:'+str(test_acc_es))