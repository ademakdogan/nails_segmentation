#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: A.Akdogan
"""

import os
from dataset_builder import NailDataset

import segmentation_models_pytorch as smp
from utils import *
from helper import utils
from torch.utils.data import DataLoader
import torch
import warnings
import pickle
import json
warnings.filterwarnings("ignore")


class Trainer:

    def __init__(self, select_class_rgb_values, class_names, device):

        self.select_class_rgb_values = select_class_rgb_values
        self.class_names = class_names
        self.device = device
        self.is_cont = True
        config_path = get_final_path(1, ['config.json'])
        with open(config_path, 'r') as f: self.config = json.load(f)
        #self.device = torch.device("mps" if torch.cuda.is_available() else "cpu")
        processed_base_data_path = get_final_path(1, ['dataset', 'processed'])
        self.x_train_dir = os.path.join(processed_base_data_path, 'train')
        self.y_train_dir = os.path.join(processed_base_data_path, 'train_labels')

        self.x_valid_dir = os.path.join(processed_base_data_path, 'val')
        self.y_valid_dir = os.path.join(processed_base_data_path, 'val_labels')

    def get_model(self):

        ENCODER = 'resnet101'
        ENCODER_WEIGHTS = 'imagenet'
        CLASSES = self.class_names
        ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation

        # create segmentation model with pretrained encoder
        model = smp.DeepLabV3Plus(
            encoder_name=ENCODER, 
            encoder_weights=ENCODER_WEIGHTS, 
            classes=len(CLASSES), 
            activation=ACTIVATION,
        )

        preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
        preprocessing_fn_path = get_final_path(1, ['model', 'preprocessing_fn.pkl'])
        pickle.dump(preprocessing_fn, open(preprocessing_fn_path, 'wb'))

        return model, preprocessing_fn
    
    def get_datasets(self, preprocessing_fn):

        train_dataset = NailDataset(self.x_train_dir, self.y_train_dir, augmentation = get_training_augmentation(), preprocessing = get_preprocessing(preprocessing_fn), class_rgb_values = self.select_class_rgb_values)
        valid_dataset = NailDataset(self.x_valid_dir, self.y_valid_dir, augmentation = get_validation_augmentation(), preprocessing = get_preprocessing(preprocessing_fn), class_rgb_values = self.select_class_rgb_values)

        return train_dataset, valid_dataset


    def train(self, model, train_dataset, valid_dataset):

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
        valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0)

        TRAINING = True

        # Set num of epochs
        EPOCHS = self.config['epochs']


        # define loss function
        #loss = smp.utils.losses.DiceLoss()
        loss = utils.losses.DiceLoss()

        # define metrics
        #metrics = [smp.utils.metrics.IoU(threshold=0.5),]

        metrics = [
            utils.metrics.IoU(threshold=self.config['iou_threshold']),
        ]

        # define optimizer
        optimizer = torch.optim.Adam([ 
            dict(params=model.parameters(), lr=self.config['lr']),
        ])

        # define learning rate scheduler (not used in this NB)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=self.config['t_0'], T_mult=self.config['t_mult'], eta_min=self.config['eta_min'],
        )

        best_model_path = get_final_path(1, ['model', 'best_model.pth'])
        # load best saved model checkpoint from previous commit (if present)
        if os.path.exists(best_model_path) and self.is_cont == True:
            model = torch.load(best_model_path, map_location=self.device)
        
        train_epoch = utils.train.TrainEpoch(
                                            model, 
                                            loss=loss,
                                            metrics=metrics, 
                                            optimizer=optimizer,
                                            device=self.device,
                                            #device='mps',
                                            verbose=True,
        )

        valid_epoch = utils.train.ValidEpoch(
                                        model, 
                                        loss=loss, 
                                        metrics=metrics, 
                                        device=self.device,
                                        #device='mps',
                                        verbose=True,
        )


        if TRAINING:

            best_iou_score = 0.0
            train_logs_list, valid_logs_list = [], []

            for i in range(0, EPOCHS):

                # Perform training & validation
                print('\nEpoch: {}'.format(i))
                train_logs = train_epoch.run(train_loader)
                valid_logs = valid_epoch.run(valid_loader)
                train_logs_list.append(train_logs)
                valid_logs_list.append(valid_logs)

                # Save model if a better val IoU score is obtained
                if best_iou_score < valid_logs['iou_score']:
                    best_iou_score = valid_logs['iou_score']
                    torch.save(model, best_model_path)
                    print('Model saved!')

    def main(self):

        model, preprocessing_fn = self.get_model()
        train_dataset, valid_dataset = self.get_datasets(preprocessing_fn)
        self.train(model, train_dataset, valid_dataset)

        return



