# -*- coding: utf-8 -*-
"""
Created on Tue May 29 15:32:45 2018

@author: Md. Kamrul Hasan
"""
print(__doc__)
# import necessary modules
import time
startTime = time.time()
import numpy as np
from pandas import DataFrame
np.random.seed(2017)
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing import image
from keras.models import Model
import cv2
import glob

#%%----------------------------------------------------
# Loading the training data
imageTrainPositive=sorted(glob.glob("C:\\Users\\Md. Kamrul Hasan\\Desktop\\PR Project Done By Md. Kamrul Hasan\\Code and Materials\\Small\\MAMMO_TRAIN_DIR\\Positive\\*.tif"))

ImageTrainPos=[]
for img in imageTrainPositive:
    cv_img_GT = cv2.imread(img)
    cv_img_GT = cv2.resize(cv_img_GT,(224, 224), interpolation = cv2.INTER_AREA)
    ImageTrainPos.append(cv_img_GT)

lenghtTP=len(ImageTrainPos)
base_model = VGG19(weights='imagenet')

Features_array = np.zeros(shape=(lenghtTP,4096))

for ind in range(lenghtTP):
    img = ImageTrainPos[ind]
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    model = Model(input=base_model.input, output=base_model.get_layer('fc2').output)
    Features_array[ind,:] = model.predict(img)
df = DataFrame( Features_array,index=None)
df.to_csv('NameTrainPOS.csv')

#%%----------------------------------------------------
imageTrainNegative=sorted(glob.glob("C:\\Users\\Md. Kamrul Hasan\\Desktop\\PR Project Done By Md. Kamrul Hasan\\Code and Materials\\Small\\MAMMO_TRAIN_DIR\\Negative\\*.tif"))

ImageTrainNeg=[]
for img in imageTrainNegative:
    cv_img_GT = cv2.imread(img)
    cv_img_GT = cv2.resize(cv_img_GT,(224, 224), interpolation = cv2.INTER_AREA)
    ImageTrainNeg.append(cv_img_GT)

lenghtTN=len(ImageTrainNeg)
base_model = VGG19(weights='imagenet')

Features_array = np.zeros(shape=(lenghtTN,4096))

for ind in range(lenghtTN):
    img = ImageTrainNeg[ind]
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    model = Model(input=base_model.input, output=base_model.get_layer('fc2').output)
    Features_array[ind,:] = model.predict(img)
df = DataFrame( Features_array,index=None)
df.to_csv('NameTrainNEG.csv')

#%%----------------------------------------------------
imageTestNegative=sorted(glob.glob("C:\\Users\\Md. Kamrul Hasan\\Desktop\\PR Project Done By Md. Kamrul Hasan\\Code and Materials\\Small\\MAMMO_TEST_DIR\\Negative\\*.tif"))

ImageTestNeg=[]
for img in imageTestNegative:
    cv_img_GT = cv2.imread(img)
    cv_img_GT = cv2.resize(cv_img_GT,(224, 224), interpolation = cv2.INTER_AREA)
    ImageTestNeg.append(cv_img_GT)

lenghtTeN=len(ImageTestNeg)
base_model = VGG19(weights='imagenet')

Features_array = np.zeros(shape=(lenghtTeN,4096))

for ind in range(lenghtTeN):
    img = ImageTestNeg[ind]
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    model = Model(input=base_model.input, output=base_model.get_layer('fc2').output)
    Features_array[ind,:] = model.predict(img)
df = DataFrame( Features_array,index=None)
df.to_csv('NameTestNeG.csv')

#%%----------------------------------------------------
imageTestPositive=sorted(glob.glob("C:\\Users\\Md. Kamrul Hasan\\Desktop\\PR Project Done By Md. Kamrul Hasan\\Code and Materials\\Small\\MAMMO_TEST_DIR\\Positive\\*.tif"))

ImageTestPos=[]
for img in imageTestPositive:
    cv_img_GT = cv2.imread(img)
    cv_img_GT = cv2.resize(cv_img_GT,(224, 224), interpolation = cv2.INTER_AREA)
    ImageTestPos.append(cv_img_GT)

lenghtTeP=len(ImageTestPos)
base_model = VGG19(weights='imagenet')

Features_array = np.zeros(shape=(lenghtTeP,4096))

for ind in range(lenghtTeP):
    img = ImageTestPos[ind]
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    model = Model(input=base_model.input, output=base_model.get_layer('fc2').output)
    Features_array[ind,:] = model.predict(img)
df = DataFrame( Features_array,index=None)
df.to_csv('NameTestPOS.csv')

#%%----------------------------------------------------
endTime = time.time()
print('It took {0:0.1f} seconds'.format(endTime - startTime))