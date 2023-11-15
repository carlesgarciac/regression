import torchio as tio

import torch

from torch.utils.data import DataLoader

from data.dataset_4D_MNMS_SA import get_dataset_SA
from data.dataset_2D_MNMS_LA import get_dataset_LA

from pathlib import Path

import numpy as np


def loader_manager(data_view, batch_size, num_workers, create_landmarks=False):

    if data_view == 'SA':
        train_set = Path('/home/carlesgc/Projects/regression/train_data_regression/')
        val_set = Path('/home/carlesgc/Projects/regression/val_data_regression/')
        test_set = Path('/home/carlesgc/Projects/regression/test_data_regression/')
        get_dataset = get_dataset_SA
        histogram_landmarks_path = '/home/carlesgc/Projects/regression/landmarks_SA_regression.npy'
        if create_landmarks == False:
            landmarks = np.load(histogram_landmarks_path)
    # elif data_view == 'LA':
    #     train_set = Path('/home/carlesgc/Projects/segmentation/train_data_LA/')
    #     get_dataset = get_dataset_LA
    #     histogram_landmarks_path = '/home/carlesgc/Projects/segmentation/landmarks_LA.npy'
    #     if create_landmarks == False:
    #         landmarks = np.load(histogram_landmarks_path)

    ## Datasets
    # train_set = Path('/home/carlesgc/Projects/segmentation/train_data_SA/')
    images_dir_train = train_set / 'images'
    labels_dir_train = train_set / 'labels'

    images_dir_val = val_set / 'images'
    labels_dir_val = val_set / 'labels'

    images_dir_test = test_set / 'images'
    labels_dir_test = test_set / 'labels'

    image_paths = sorted(images_dir_train.glob('*.nii.gz'))

    if create_landmarks == True:
        landmarks = tio.HistogramStandardization.train(
            image_paths,
            output_path=histogram_landmarks_path,
        )

    training_transform = tio.Compose([
        tio.CropOrPad((256,256,15)),
        # tio.ZNormalization(masking_method=tio.ZNormalization.mean),
        # tio.HistogramStandardization({'mri': landmarks}),
        tio.RescaleIntensity(out_min_max=(0,1)),
        # tio.OneHot()
    ])
    validation_transform = tio.Compose([
        tio.CropOrPad((256,256,15)),
        # tio.ZNormalization(masking_method=tio.ZNormalization.mean),
        # tio.HistogramStandardization({'mri': landmarks}),
        tio.RescaleIntensity(out_min_max=(0,1)),
        # tio.OneHot()
    ])
    test_transform = tio.Compose([
        tio.CropOrPad((256,256,15)),
        # tio.ZNormalization(masking_method=tio.ZNormalization.mean),
        # tio.HistogramStandardization({'mri': landmarks}),
        tio.RescaleIntensity(out_min_max=(0,1)),
        # tio.OneHot()
    ])

    subjects_train = get_dataset(images_dir_train, labels_dir_train)
    subjects_val = get_dataset(images_dir_val, labels_dir_val)
    subjects_test = get_dataset(images_dir_test, labels_dir_test)
    # num_subjects = len(subjects)
    # num_training_subjects = int(training_split_ratio * num_subjects)
    # num_validation_subjects = num_subjects - num_training_subjects

    # num_split_subjects = num_training_subjects, num_validation_subjects
    # training_subjects, validation_subjects = torch.utils.data.random_split(subjects, num_split_subjects)

    training_subjects = subjects_train
    validation_subjects = subjects_val
    testing_subjects = subjects_test

    training_set = tio.SubjectsDataset(
        training_subjects, transform=training_transform)

    validation_set = tio.SubjectsDataset(
        validation_subjects, transform=validation_transform)
    
    testing_set = tio.SubjectsDataset(
        testing_subjects, transform=test_transform)
    print('Training set:', len(training_set), 'subjects')
    print('Validation set:', len(validation_set), 'subjects')
    print('Test set:', len(testing_set), 'subjects')

    ## Loaders
    train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(validation_set, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(testing_set, batch_size=batch_size, num_workers=num_workers)

    return train_loader, val_loader, test_loader
