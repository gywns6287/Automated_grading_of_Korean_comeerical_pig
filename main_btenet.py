#python main_btenet.py --img data\head_img --out btenet_results --device 0

import os
import numpy as np
from utils import *
from model import *

args = parser()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]=args.device

# Load model
model = call_BTENet()
model.summary()

#Test 
model.load_weights(os.path.join('weights','btenet.h5'))

test_files = os.listdir(args.img)
test_X = head_loader(args.img)


predictor = inference_BTENet(model, test_X)

btenet_save(test_files, predictor, args.out)
paint_btenet(test_files, test_X, predictor, args.img,args.out)