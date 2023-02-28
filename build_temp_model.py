# -*- coding: utf-8 -*-

import numpy as np


def build_cnn_model():
    model_weight = [np.zeros(shape=(5,5,1,64),dtype=np.float32),
                  np.zeros(shape=(64),dtype=np.float32),
                  np.zeros(shape=(3,3,64,128),dtype=np.float32),
                  np.zeros(shape=(128),dtype=np.float32),
                  np.zeros(shape=(6272,256),dtype=np.float32),
                  np.zeros(shape=(256),dtype=np.float32),
                  np.zeros(shape=(256,64),dtype=np.float32),
                  np.zeros(shape=(64),dtype=np.float32),
                  np.zeros(shape=(64,10),dtype=np.float32),
                  np.zeros(shape=(10),dtype=np.float32)]
    model_len = len(model_weight)
    return model_weight, model_len


def build_logistic_model():
    model_weight = [np.zeros(shape=(784,10),dtype=np.float32),
                         np.zeros(shape=(10),dtype=np.float32)]
    model_len = len(model_weight)
    return model_weight, model_len


def build_resnet_model():
    model_weight = [np.zeros(shape=(7,7,1,64),dtype=np.float32),
                     
                  np.zeros(shape=(64),dtype=np.float32),
                  np.zeros(shape=(64),dtype=np.float32),
                  np.zeros(shape=(64),dtype=np.float32),
                  np.zeros(shape=(64),dtype=np.float32),
                  np.zeros(shape=(64),dtype=np.float32),
                  np.zeros(shape=(3,3,64,64),dtype=np.float32),
                  
                  np.zeros(shape=(64),dtype=np.float32),
                  np.zeros(shape=(64),dtype=np.float32),
                  np.zeros(shape=(64),dtype=np.float32),
                  np.zeros(shape=(64),dtype=np.float32),
                  np.zeros(shape=(64),dtype=np.float32),
                  np.zeros(shape=(3,3,64,64),dtype=np.float32),
                  
                  np.zeros(shape=(64),dtype=np.float32),
                  np.zeros(shape=(64),dtype=np.float32),
                  np.zeros(shape=(64),dtype=np.float32),
                  np.zeros(shape=(64),dtype=np.float32),
                  np.zeros(shape=(64),dtype=np.float32),
                  np.zeros(shape=(3,3,64,64),dtype=np.float32),
                  
                  np.zeros(shape=(64),dtype=np.float32),
                  np.zeros(shape=(64),dtype=np.float32),
                  np.zeros(shape=(64),dtype=np.float32),
                  np.zeros(shape=(64),dtype=np.float32),
                  np.zeros(shape=(64),dtype=np.float32),
                  np.zeros(shape=(3,3,64,64),dtype=np.float32),
                  
                  np.zeros(shape=(64),dtype=np.float32),
                  np.zeros(shape=(64),dtype=np.float32),
                  np.zeros(shape=(64),dtype=np.float32),
                  np.zeros(shape=(64),dtype=np.float32),
                  np.zeros(shape=(64),dtype=np.float32),
                  np.zeros(shape=(3,3,64,128),dtype=np.float32),
                  
                  np.zeros(shape=(128),dtype=np.float32),
                  np.zeros(shape=(128),dtype=np.float32),
                  np.zeros(shape=(128),dtype=np.float32),
                  np.zeros(shape=(128),dtype=np.float32),
                  np.zeros(shape=(128),dtype=np.float32),
                  np.zeros(shape=(3,3,128,128),dtype=np.float32),
                  
                  np.zeros(shape=(128),dtype=np.float32),
                  np.zeros(shape=(3,3,64,128),dtype=np.float32),
                  
                  np.zeros(shape=(128),dtype=np.float32),
                  np.zeros(shape=(128),dtype=np.float32),
                  np.zeros(shape=(128),dtype=np.float32),
                  np.zeros(shape=(128),dtype=np.float32),
                  np.zeros(shape=(128),dtype=np.float32),
                  np.zeros(shape=(128),dtype=np.float32),
                  np.zeros(shape=(128),dtype=np.float32),
                  np.zeros(shape=(128),dtype=np.float32),
                  np.zeros(shape=(128),dtype=np.float32),
                  
                  np.zeros(shape=(3,3,128,128),dtype=np.float32),
                  
                  np.zeros(shape=(128),dtype=np.float32),
                  np.zeros(shape=(128),dtype=np.float32),
                  np.zeros(shape=(128),dtype=np.float32),
                  np.zeros(shape=(128),dtype=np.float32),
                  np.zeros(shape=(128),dtype=np.float32),
                  np.zeros(shape=(3,3,128,128),dtype=np.float32),
                  
                  np.zeros(shape=(128),dtype=np.float32),
                  np.zeros(shape=(128),dtype=np.float32),
                  np.zeros(shape=(128),dtype=np.float32),
                  np.zeros(shape=(128),dtype=np.float32),
                  np.zeros(shape=(128),dtype=np.float32),
                  np.zeros(shape=(3,3,128,256),dtype=np.float32),
                  
                  np.zeros(shape=(256),dtype=np.float32),
                  np.zeros(shape=(256),dtype=np.float32),
                  np.zeros(shape=(256),dtype=np.float32),
                  np.zeros(shape=(256),dtype=np.float32),
                  np.zeros(shape=(256),dtype=np.float32),
                  np.zeros(shape=(3,3,256,256),dtype=np.float32),
                  
                  np.zeros(shape=(256),dtype=np.float32),
                  np.zeros(shape=(3,3,128,256),dtype=np.float32),
                  
                  np.zeros(shape=(256),dtype=np.float32),
                  np.zeros(shape=(256),dtype=np.float32),
                  np.zeros(shape=(256),dtype=np.float32),
                  np.zeros(shape=(256),dtype=np.float32),
                  np.zeros(shape=(256),dtype=np.float32),
                  np.zeros(shape=(256),dtype=np.float32),
                  np.zeros(shape=(256),dtype=np.float32),
                  np.zeros(shape=(256),dtype=np.float32),
                  np.zeros(shape=(256),dtype=np.float32),
                  np.zeros(shape=(3,3,256,256),dtype=np.float32),
                  
                  np.zeros(shape=(256),dtype=np.float32),
                  np.zeros(shape=(256),dtype=np.float32),
                  np.zeros(shape=(256),dtype=np.float32),
                  np.zeros(shape=(256),dtype=np.float32),
                  np.zeros(shape=(256),dtype=np.float32),
                  np.zeros(shape=(3,3,256,256),dtype=np.float32),
                  
                  np.zeros(shape=(256),dtype=np.float32),
                  np.zeros(shape=(256),dtype=np.float32),
                  np.zeros(shape=(256),dtype=np.float32),
                  np.zeros(shape=(256),dtype=np.float32),
                  np.zeros(shape=(256),dtype=np.float32),
                  np.zeros(shape=(3,3,256,512),dtype=np.float32),
                  
                  np.zeros(shape=(512),dtype=np.float32),
                  np.zeros(shape=(512),dtype=np.float32),
                  np.zeros(shape=(512),dtype=np.float32),
                  np.zeros(shape=(512),dtype=np.float32),
                  np.zeros(shape=(512),dtype=np.float32),
                  np.zeros(shape=(3,3,512,512),dtype=np.float32),
                  
                  np.zeros(shape=(512),dtype=np.float32),
                  np.zeros(shape=(3,3,256,512),dtype=np.float32),
                  
                  np.zeros(shape=(512),dtype=np.float32),
                  np.zeros(shape=(512),dtype=np.float32),
                  np.zeros(shape=(512),dtype=np.float32),
                  np.zeros(shape=(512),dtype=np.float32),
                  np.zeros(shape=(512),dtype=np.float32),
                  np.zeros(shape=(512),dtype=np.float32),
                  np.zeros(shape=(512),dtype=np.float32),
                  np.zeros(shape=(512),dtype=np.float32),
                  np.zeros(shape=(512),dtype=np.float32),
                  
                  np.zeros(shape=(3,3,512,512),dtype=np.float32),
                  
                  np.zeros(shape=(512),dtype=np.float32),
                  np.zeros(shape=(512),dtype=np.float32),
                  np.zeros(shape=(512),dtype=np.float32),
                  np.zeros(shape=(512),dtype=np.float32),
                  np.zeros(shape=(512),dtype=np.float32),
                  np.zeros(shape=(3,3,512,512),dtype=np.float32),
                  
                  np.zeros(shape=(512),dtype=np.float32),
                  np.zeros(shape=(512),dtype=np.float32),
                  np.zeros(shape=(512),dtype=np.float32),
                  np.zeros(shape=(512),dtype=np.float32),
                  np.zeros(shape=(512),dtype=np.float32),
                  
                  np.zeros(shape=(512,10),dtype=np.float32),
                  np.zeros(shape=(10),dtype=np.float32)]

    model_len = len(model_weight)
    return model_weight, model_len