#python main_scnet.py --img data\hip_img --out scnet_results

import numpy as np
from utils import *
from model import *
import os

args = parser()
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]=args.device


# Load model
model = Load_SCNet(3, base_model = 'ResNet', input_shape = (224,224,3))
model.summary()

#Inference
model.load_weights(os.path.join('weights','scnet.h5'))

test_files = os.listdir(args.img)
test_X = heap_loader(args.img)

predictor = inference_SCNet(model, test_X)

scnet_save(test_files, predictor.pre_class, predictor.prob, args.out)
paint_scnet(test_files, test_X, predictor,args.img, args.out)