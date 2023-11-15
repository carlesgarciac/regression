from numpy import dtype, uint8
from torchvision import transforms

import torchio as tio

import napari

import numpy as np


def dataloader_checker(train_loader):
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx == 7:
                image = batch['mri'][tio.DATA]
                # print(image.shape)
                image = image[0,:,:,:,:]
                
                # with napari.gui_qt():
                image = np.array(image*255, dtype=uint8)
                image = np.moveaxis(image, -1, 1)
                # print(image.shape)
                viewer = napari.Viewer()
                viewer.add_image(image, name='test')
                napari.run()

        # print(image.shape)  
        # img = np.transpose(image, (1,2,0))
        # image = image*255/np.max(image)
        # img = transforms.ToPILImage()(image)
        # img = img*255/np.max(img)
        # img.save('debug.png')