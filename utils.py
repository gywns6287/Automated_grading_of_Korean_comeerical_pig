import os

import tqdm
import cv2
import numpy as np
from PIL import Image
import argparse

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img")
    parser.add_argument("--out",default='.')
    parser.add_argument("--device", default = '0')
    return parser.parse_args()

def img_crop(img,crop_point):
    h, w, _ = img.shape
    ch = int(h*(1-crop_point[0]))
    cw = int(w*(1-crop_point[1]))
    return img[ch:,cw:,:] 

def text_to_numpy(path,sep='\t'):
    data = []
    for line in open(path):
        data.append(line.strip().split('\t'))
    return np.array(data)

def heap_loader(
    path, 
    crop_point=(0.6,0.6),
    input_size = (224,224)):
    
    imgs = os.listdir(path)
    X = np.zeros((len(imgs),input_size[0],input_size[1],3))
    
    for idx, i in tqdm.tqdm(enumerate(imgs), total = len(imgs)):

        img = np.array(Image.open(os.path.join(path,i)))
        img = img_crop(img, crop_point)
        X[idx,...] = cv2.resize(img , input_size) / 255.
    
    return X


def head_loader(
    img_path,
    crop = 75,
    input_shape = (574,768)):
    
    imgs = os.listdir(img_path)

    X = np.zeros((len(imgs),input_shape[0],crop,3))
    
    for idx, i in tqdm.tqdm(enumerate(imgs), total = len(imgs)):

        img = np.array(Image.open(os.path.join(img_path,i)))
        img = img[:,:crop,:]
        X[idx,...] = img/255.

    return X

def scnet_save(filenames, pred, prob, out):
    with open(out+'/'+'pred.sol','w') as sol:
        sol.write('FILE\tPRED\tprob\n')
        for f,  p, pr in zip(filenames,pred, prob):
            sol.write('\t'.join([f, p, str(pr)])+'\n')

def btenet_save(filenames, predictor, out):
    with open(out+'/'+'bf.sol','w') as sol:
        sol.write('FILE\tPRED\n')
        for f, p in zip(filenames, predictor.pred_bf):
            sol.write('\t'.join([f, str(p)])+'\n')

def paint(img,mask,color,alpha=0.3):
    
    label = mask >= 127

    for c in range(3):
        img[:, :, c] = np.where(label,
                                img[:, :, c] * (1 - alpha) + alpha * color[c],
                                img[:, :, c])
        
    _, thresh = cv2.threshold(mask, 127, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img,contours,-1,color,2)
        
    return img    

def paint_btenet(filenames, X, predictor, img_path,out):

    X = (X*255).astype(np.uint8)
    masks = (predictor.pred_mask*255).astype(np.uint8)
    os.system('mkdir "{0}/mask" "{0}/paint"'.format(out))
    
    for x, mask, name,pred in tqdm.tqdm(zip(X,masks,filenames,predictor.pred_bf),total = len(filenames)):
    
        cv2.imwrite('{0}/mask/{1}'.format(out,name),mask)
        x = paint(x, mask[...,0], (51,255,0))
        img = cv2.imread(os.path.join(img_path,name))
        img[:,:75,:] = x[...,::-1]

        img = cv2.putText(img, 'Pred: '+str(round(pred,1))+' mm', (280,60), cv2.FONT_HERSHEY_SIMPLEX, 2,
		       (0,255,0),6)

        cv2.imwrite('{0}/paint/{1}'.format(out,name),img)
        


def paint_scnet(filenames, X, predictor,img_path, out):

    X = (X*255).astype(np.uint8)
    os.system('mkdir "{0}/paint"'.format(out))
    
    for name, pred,prob in tqdm.tqdm(zip(filenames,predictor.pre_class,predictor.prob),total = len(filenames)):
        
        prob = str(round(float(prob),3))  
        if len(prob) < 5:
            prob += '0'*(5 - len(prob))
        
        img = np.array(Image.open(os.path.join(img_path,name)))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.putText(img, pred, (300,90), cv2.FONT_HERSHEY_SIMPLEX, 4, (0,255,0),8) 
        img = cv2.putText(img, '('+str(prob)+')', (300,200), cv2.FONT_HERSHEY_SIMPLEX, 4, (0,255,0),8)

        cv2.imwrite('{0}/paint/{1}'.format(out, name),img)
        
