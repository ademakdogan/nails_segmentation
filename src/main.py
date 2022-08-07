#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: A.Akdogan
"""

import pandas as pd
import numpy as np
from trainer import Trainer
from evaluator import Evaluator
from predictor import Predictor
from utils import *
import argparse





if __name__ == '__main__':

    class_dict_path =  get_final_path(1, ['labels', 'label_class_dict.csv'])
    class_dict = pd.read_csv(class_dict_path)
    class_names = class_dict['name'].tolist()
    class_rgb_values = class_dict[['r','g','b']].values.tolist()
    select_classes = ['background', 'nail']
    select_class_indices = [class_names.index(cls.lower()) for cls in select_classes]
    select_class_rgb_values =  np.array(class_rgb_values)[select_class_indices]

    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--mode", required = True, help = 'train, predict or evaluate')
    ap.add_argument("-d", "--device", required = True, help = 'mps, gpu or cpu')
    ap.add_argument("-i", "--img_path", required = False, default = '', help = 'image path for mode of predict')

    args = vars(ap.parse_args())

    if args['mode'] == 'train':
        Trainer(select_class_rgb_values, class_names, args['device']).main()
    elif args['mode'] == 'predict':
        Predictor(args['img_path'], select_class_rgb_values, select_classes, device = args['device']).main()
    elif args['mode'] == 'evaluate':
        Evaluator(select_class_rgb_values, device = args['device']).evaluate()
    else: 
        raise Exception('train, predict or evaluate can be used as mode')
