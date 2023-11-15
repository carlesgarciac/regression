import torch
import torchvision
import torchio as tio
import torch.nn.functional as F
import pandas as pd
from pathlib import Path

import numpy as np

def get_dataset_SA(image_dir, label_dir):
    sheet = pd.read_csv('/home/carlesgc/Projects/regression/MnM/201014_M&Ms_Dataset_Information_-_opendataset.csv', index_col='External code')

    image_paths = sorted(image_dir.glob('*.nii.gz'))
    label_paths = sorted(label_dir.glob('*.nii.gz'))
    assert len(image_paths) == len(label_paths)

    subjects = []
    for (image_path, label_path) in zip(image_paths, label_paths):
        patient = str(image_path).split('/')[-1].split('_')[0]
        subject = tio.Subject(
            mri=tio.ScalarImage(image_path),
            heart=tio.LabelMap(label_path),
            ED=sheet.loc[patient].loc["ED"],
            ES=sheet.loc[patient].loc["ES"],
            patient=patient
        )
        # if subject['mri'][tio.DATA].shape[1] > 35:
        #     continue
        subjects.append(subject)
    # dataset = tio.SubjectsDataset(subjects)
    # print('Dataset size:', len(dataset), 'subjects')
    return subjects

# sheet = pd.read_csv('/home/carlesgc/Projects/regression/MnM/201014_M&Ms_Dataset_Information_-_opendataset.csv', index_col='External code')


# train_set = Path('/home/carlesgc/Projects/regression/train_data_regression/')

# image_dir = train_set / 'images'
# label_dir = train_set / 'labels'

# image_paths = sorted(image_dir.glob('*.nii.gz'))
# label_paths = sorted(label_dir.glob('*.nii.gz'))
# assert len(image_paths) == len(label_paths)

# subjects = []
# for (image_path, label_path) in zip(image_paths, label_paths):
#     # print(image_path)
#     patient = str(image_path).split('/')[-1].split('_')[0]
#     print(sheet.loc[patient].loc["ES"])
#     subject = tio.Subject(
#         mri=tio.ScalarImage(image_path),
#         heart=tio.LabelMap(label_path),
#         ED=sheet.loc[patient].loc["ED"],
#         ES=sheet.loc[patient].loc["ES"]
#     )
#     subjects.append(subject)
# # dataset = tio.SubjectsDataset(subjects)
# # print('Dataset size:', len(dataset), 'subjects')