#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: A.Akdogan
"""
from utils import *
from data_visualizer import *
from dataset_builder import NailDataset
from trainer import Trainer
import os
from torch.utils.data import DataLoader
import pickle
import cv2
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class SingleNail(torch.utils.data.Dataset):
    
    def __init__(
            self, 
            img_path, 
            class_rgb_values=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        
        self.image_paths = [img_path]
        self.class_rgb_values = class_rgb_values
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        image = cv2.cvtColor(cv2.imread(self.image_paths[i]), cv2.COLOR_BGR2RGB)
        #-----
        border_v = 0
        border_h = 0
        IMG_COL = 800
        IMG_ROW = 800
        if (IMG_COL/IMG_ROW) >= (image.shape[0]/image.shape[1]):
            border_v = int((((IMG_COL/IMG_ROW)*image.shape[1])-image.shape[0])/2)
        else:
            border_h = int((((IMG_ROW/IMG_COL)*image.shape[0])-image.shape[1])/2)
        image = cv2.copyMakeBorder(image, border_v, border_v, border_h, border_h, cv2.BORDER_CONSTANT, 0)
        image = cv2.resize(image, (IMG_ROW, IMG_COL))
        #-----
        # apply augmentations
        if self.augmentation: image = self.augmentation(image=image)['image']
        
        # apply preprocessing
        if self.preprocessing: image = self.preprocessing(image=image)['image']

        return image

class Predictor:

    def __init__(self, img_path, select_class_rgb_values, select_classes, device):

        preprocessing_fn_path = get_final_path(1, ['model', 'preprocessing_fn.pkl'])
        model_path = get_final_path(1, ['model', 'best_model.pth'])
        self.model = torch.load(model_path, map_location = device)
        self.preprocessing_fn = pickle.load(open(preprocessing_fn_path, 'rb'))
        self.select_classes = select_classes
        self.img_path = img_path
        self.select_class_rgb_values = select_class_rgb_values
        self.device = device
        
        processed_base_data_path = get_final_path(1, ['dataset', 'processed'])
        self.x_test_dir = os.path.join(processed_base_data_path, 'test')
        self.y_test_dir = os.path.join(processed_base_data_path, 'test_labels')

    @staticmethod
    def crop_image(image, target_image_dims=[800,800,3]):
    
        target_size = target_image_dims[0]
        image_size = len(image)
        padding = (image_size - target_size) // 2

        return image[
            padding:image_size - padding,
            padding:image_size - padding,
            :,
        ]

    def img_preprocess(self):

        return SingleNail(self.img_path, augmentation = get_validation_augmentation(), preprocessing = get_preprocessing(self.preprocessing_fn), class_rgb_values = self.select_class_rgb_values )[0]

    def get_predicted_mask(self, processed_img):

        x_tensor = torch.from_numpy(processed_img).to(self.device).unsqueeze(0)
        # Predict test image
        pred_mask = self.model(x_tensor)
        pred_mask = pred_mask.detach().squeeze().cpu().numpy()
        pred_mask = np.transpose(pred_mask,(1,2,0))
        
        return pred_mask

    def get_visualize(self, pred_mask):

        # Get prediction channel corresponding to building
        pred_building_heatmap = pred_mask[:,:,self.select_classes.index('nail')]
        image_vis = SingleNail(self.img_path, augmentation = get_validation_augmentation(), class_rgb_values = self.select_class_rgb_values )[0]
        image_vis = Predictor.crop_image(image_vis.astype('uint8'))
        pred_mask = Predictor.crop_image(colour_code_segmentation(reverse_one_hot(pred_mask), self.select_class_rgb_values))
        # Convert gt_mask from `CHW` format to `HWC` format
        non_black_pixels_mask = np.any(pred_mask != [0, 0, 0], axis=-1)
        base_image = Image.fromarray(image_vis, 'RGB').convert("RGBA")
        mask = (np.zeros((base_image.size[1], base_image.size[0]))).astype(np.uint8)
        mask[non_black_pixels_mask] = 126
        mask = Image.fromarray(mask, mode='L')
        overlay = Image.new('RGBA', base_image.size, (255,255,255,0))
        drawing = ImageDraw.Draw(overlay)
        drawing.bitmap((0, 0), mask, fill=(10, 200, 0, 200))

        image = Image.alpha_composite(base_image, overlay)
        pth = get_final_path(1, ['dataset', 'result.png'])
        #cv2.imwrite(pth, image)
        image.save(pth)
        
        visualize(
            masked_image = image,
            #ground_truth_mask = gt_mask,
            predicted_mask = pred_mask,
            predicted_building_heatmap = pred_building_heatmap
        )

    def main(self):

        processed_img = self.img_preprocess()
        pred_mask = self.get_predicted_mask(processed_img)
        self.get_visualize(pred_mask)

        return
    






