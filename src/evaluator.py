#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: A.Akdogan
"""

from utils import *
import torch
import pickle
from helper import utils
from dataset_builder import NailDataset
from torch.utils.data import DataLoader
import os

class Evaluator: 

    def __init__(self, select_class_rgb_values, device):

        preprocessing_fn_path = get_final_path(1, ['model', 'preprocessing_fn.pkl'])
        model_path = get_final_path(1, ['model', 'best_model.pth'])
        processed_base_data_path = get_final_path(1, ['dataset', 'processed'])
        try:
            path = '/Users/adem/Documents/4-Projeler/74-Segmentation/nail_dataset/new_images/val/.DS_Store'
            os.remove(os.path.join(processed_base_data_path, 'test', '.DS_Store'))
            os.remove(os.path.join(processed_base_data_path, 'test_labels', '.DS_Store'))
        except:
            pass
        self.x_test_dir = os.path.join(processed_base_data_path, 'test')
        self.y_test_dir = os.path.join(processed_base_data_path, 'test_labels')
        self.model = torch.load(model_path, map_location = device)
        self.preprocessing_fn = pickle.load(open(preprocessing_fn_path, 'rb'))
        self.select_class_rgb_values = select_class_rgb_values
        self.device = device

    def evaluate(self):

        loss = utils.losses.DiceLoss()

        # define metrics
        #metrics = [smp.utils.metrics.IoU(threshold=0.5),]

        metrics = [
            utils.metrics.IoU(threshold=0.5),
        ]


        test_dataset = NailDataset(
        self.x_test_dir, 
        self.y_test_dir, 
        augmentation=get_validation_augmentation(), 
        preprocessing=get_preprocessing(self.preprocessing_fn),
        class_rgb_values=self.select_class_rgb_values,
        )

        test_dataloader = DataLoader(test_dataset)

        test_epoch = utils.train.ValidEpoch(
        self.model,
        loss=loss, 
        metrics=metrics, 
        device = self.device,
        verbose=True,
        )

        valid_logs = test_epoch.run(test_dataloader)
        print("Evaluation on Test Data: ")
        print(f"Mean IoU Score: {valid_logs['iou_score']:.4f}")
        print(f"Mean Dice Loss: {valid_logs['dice_loss']:.4f}")