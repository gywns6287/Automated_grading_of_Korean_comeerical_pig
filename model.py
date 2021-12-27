from tensorflow.keras.applications import ResNet50, MobileNetV2
from tensorflow.keras.layers import *
from tensorflow.keras import Model
import time
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import tqdm
import cv2

def Load_SCNet(class_num = 3, base_model = 'ResNet', input_shape = (224,224,3)):
    if base_model == 'ResNet':
        base = ResNet50(include_top = False, input_shape = input_shape)
    elif base_model == 'MobileNet':
        base = MobileNetV2(include_top= False, input_shape = input_shape)
    
    inputs = Input(input_shape)
    feature = base(inputs)
    last_cnn = Conv2D(class_num,(3,3),activation = 'relu',padding='same')(feature)
    
    GAP = GlobalAveragePooling2D()(last_cnn)
    out =  Dense(class_num,activation = 'softmax')(GAP)
    return Model(inputs,out)

class inference_SCNet():
    def __init__(self, model, X , code = {0:'female',1:'boar',2:'barrow'}):
        pre = model.predict(X, verbose=1) 
        self.pre_class = np.array([code[i] for i in np.argmax(pre, axis = 1)])
        self.prob = pre[(range(len(pre)),np.argmax(pre, axis = 1))] 



def get_unet(input_shape = (None,256,256,3),initial_filter = 64):
    inputs = Input(input_shape)
    conv1 = Conv2D(initial_filter , (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1_1 = Conv2D(initial_filter , (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPool2D(pool_size = (2,2))(conv1_1)

    conv2 = Conv2D(initial_filter * 2, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2_1 = Conv2D(initial_filter * 2, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPool2D(pool_size = (2,2))(conv2_1)

    conv3 = Conv2D(initial_filter * 4, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3_1 = Conv2D(initial_filter * 4, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPool2D(pool_size = (2,2))(conv3_1)

    conv4 = Conv2D(initial_filter * 8, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4_1 = Conv2D(initial_filter * 8, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4_1 = Dropout(0.5)(conv4_1)
    pool4 = MaxPool2D(pool_size = (2,2))(conv4_1)

    ####################################downsampling#############################################

    conv5 = Conv2D(initial_filter * 16, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5_1 = Conv2D(initial_filter * 16, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    conv5_1 = Dropout(0.5)(conv5_1)
    up5 = UpSampling2D(size = (2,2))(conv5_1)
    up5 = ZeroPadding2D(padding=((1,0),(0,1)))(up5)
    ####################################upsampling#############################################
    merge_4 = concatenate([conv4_1,up5], axis = 3)
    conv_4 = Conv2D(initial_filter * 8, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge_4)
    conv_4_1 = Conv2D(initial_filter * 8, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv_4)
    up4 = UpSampling2D(size = (2,2))(conv_4_1)
    up4 = ZeroPadding2D(padding=((0,1),(0,0)))(up4)

    merge_3 = concatenate([conv3_1,up4], axis = 3)
    conv_3 = Conv2D(initial_filter * 4, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge_3)
    conv_3_1 = Conv2D(initial_filter * 4, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv_3)
    up3 = UpSampling2D(size = (2,2))(conv_3_1)
    up3 = ZeroPadding2D(padding=((1,0),(0,1)))(up3)

    merge_2 = concatenate([conv2_1,up3], axis = 3)
    conv_2 = Conv2D(initial_filter * 2, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge_2)
    conv_2_1 = Conv2D(initial_filter * 2, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv_2)
    up2 = UpSampling2D(size = (2,2))(conv_2_1)
    up2 = ZeroPadding2D(padding=((0,0),(0,1)))(up2)

    merge_1 = concatenate([conv1_1,up2], axis = 3)
    conv_1 = Conv2D(initial_filter, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge_1)
    conv_1_1 = Conv2D(initial_filter, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv_1)

    mask = Conv2D(1, (1,1), activation = 'sigmoid',name='mask')(conv_1_1)
    
    return Model(inputs,mask)


def add_bf_module(model,freeze=False):
    if freeze:
        model.trainable = False
    unet_f = Flatten()(model.output)
    resnet_f = ResNet50(include_top=False,input_shape=(574,75,3))(model.input)
    resnet_f = GlobalAveragePooling2D()(resnet_f)

    fc = Concatenate()([unet_f,resnet_f])
    fc = Dense(1024,activation='relu')(fc)
    fc = Dense(1024,activation='relu')(fc)
    fc = Dense(1,name = 'bf')(fc)

    return Model(model.input,fc)

def call_BTENet(input_shape = (574,75,3)):
    model = get_unet(input_shape= input_shape,initial_filter = 64)
    model = add_bf_module(model,freeze=True)
    return Model(model.input,[model.layers[-8].output,model.output])

class inference_BTENet():
    def __init__(self, model, X):

        mask, bf = model.predict(X, verbose=1) 
        self.pred_mask = mask
        self.pred_bf = bf[...,0]
