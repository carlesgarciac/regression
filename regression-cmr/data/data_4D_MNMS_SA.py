import os
import nibabel as nib
import numpy as np
import pandas as pd

mode = 'train'

path_data = '/home/carlesgc/Projects/regression/MnM/Training/Labeled/'
path_out = '/home/carlesgc/Projects/regression/train_data_regression/'

path_data_val = '/home/carlesgc/Projects/regression/MnM/Validation/'
path_out_val = '/home/carlesgc/Projects/regression/val_data_regression/'

path_data_test = '/home/carlesgc/Projects/regression/MnM/Testing/'
path_out_test = '/home/carlesgc/Projects/regression/test_data_regression/'

sheet = pd.read_csv('/home/carlesgc/Projects/regression/MnM/201014_M&Ms_Dataset_Information_-_opendataset.csv')

if mode == 'train':
    for (dirpath, dirnames, filenames) in os.walk(path_data):
        for file in filenames:
            dir_patient = file.split('_')[0] 
            if 'gt' not in file:
                original = nib.load(os.path.join(path_data+dir_patient+'/', file)).get_fdata()
                affine = nib.load(os.path.join(path_data+dir_patient+'/', file)).affine
                hdr = nib.load(os.path.join(path_data+dir_patient+'/', file)).header
                out = nib.Nifti1Image(original, affine, header=hdr)
                print(os.path.join(path_out+'images/', file.split('.')[0]+'.nii.gz'))
                nib.save(out, os.path.join(path_out+'images/', file.split('.')[0]+'.nii.gz'))
            else:
                original_gt = nib.load(os.path.join(path_data+dir_patient+'/', file.split('.')[0]+'.nii.gz')).get_fdata()
                affine_gt = nib.load(os.path.join(path_data+dir_patient+'/', file.split('.')[0]+'.nii.gz')).affine
                hdr_gt = nib.load(os.path.join(path_data+dir_patient+'/', file.split('.')[0]+'.nii.gz')).header
                out_gt = nib.Nifti1Image(original_gt, affine_gt, header=hdr_gt)
                nib.save(out_gt, os.path.join(path_out+'labels/', file.split('.')[0]+'.nii.gz'))

elif mode == 'val':
    for (dirpath, dirnames, filenames) in os.walk(path_data_val):
        for file in filenames:
            dir_patient = file.split('_')[0] 
            if 'gt' not in file:
                original = nib.load(os.path.join(path_data_val+dir_patient+'/', file)).get_fdata()
                affine = nib.load(os.path.join(path_data_val+dir_patient+'/', file)).affine
                hdr = nib.load(os.path.join(path_data_val+dir_patient+'/', file)).header
                out = nib.Nifti1Image(original, affine, header=hdr)
                print(os.path.join(path_out_val+'images/', file.split('.')[0]+'.nii.gz'))
                nib.save(out, os.path.join(path_out_val+'images/', file.split('.')[0]+'.nii.gz'))
            else:
                original_gt = nib.load(os.path.join(path_data_val+dir_patient+'/', file.split('.')[0]+'.nii.gz')).get_fdata()
                affine_gt = nib.load(os.path.join(path_data_val+dir_patient+'/', file.split('.')[0]+'.nii.gz')).affine
                hdr_gt = nib.load(os.path.join(path_data_val+dir_patient+'/', file.split('.')[0]+'.nii.gz')).header
                out_gt = nib.Nifti1Image(original_gt, affine_gt, header=hdr_gt)
                nib.save(out_gt, os.path.join(path_out_val+'labels/', file.split('.')[0]+'.nii.gz'))

elif mode == 'test':
    for (dirpath, dirnames, filenames) in os.walk(path_data_test):
        for file in filenames:
            dir_patient = file.split('_')[0] 
            if 'gt' not in file:
                original = nib.load(os.path.join(path_data_test+dir_patient+'/', file)).get_fdata()
                affine = nib.load(os.path.join(path_data_test+dir_patient+'/', file)).affine
                hdr = nib.load(os.path.join(path_data_test+dir_patient+'/', file)).header
                out = nib.Nifti1Image(original, affine, header=hdr)
                print(os.path.join(path_out_test+'images/', file.split('.')[0]+'.nii.gz'))
                nib.save(out, os.path.join(path_out_test+'images/', file.split('.')[0]+'.nii.gz'))
            else:
                original_gt = nib.load(os.path.join(path_data_test+dir_patient+'/', file.split('.')[0]+'.nii.gz')).get_fdata()
                affine_gt = nib.load(os.path.join(path_data_test+dir_patient+'/', file.split('.')[0]+'.nii.gz')).affine
                hdr_gt = nib.load(os.path.join(path_data_test+dir_patient+'/', file.split('.')[0]+'.nii.gz')).header
                out_gt = nib.Nifti1Image(original_gt, affine_gt, header=hdr_gt)
                nib.save(out_gt, os.path.join(path_out_test+'labels/', file.split('.')[0]+'.nii.gz'))
