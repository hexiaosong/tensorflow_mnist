# coding: utf-8
"""
Create on 2018/10/31

@author:hexiaosong
"""
import os
import cv2
import glob
import numpy as np
from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.preprocessing.image import *

basedir = "/Users/apple/Downloads/catvsdog/"

cate_dict = {'cats':0, 'dogs':1}

print("-------- loading train data")
X_train = list()
y_train = list()
for item in ['cats','dogs']:
    dir = os.path.join(basedir, "train", item)
    image_files = glob.glob(os.path.join(dir,"*.jpg"))
    print("loding {}, image count={}".format(dir, len(image_files)))
    for image_file in image_files:
        image = cv2.imread(image_file)
        X_train.append(cv2.resize(image, (128, 128)))
        label = np.zeros(2, dtype=np.uint8)
        label[cate_dict[item]]=1
        y_train.append(label)
X_train = np.array(X_train)
y_train = np.array(y_train)

print("-------- loading valid data")
X_valid = list()
y_valid = list()
for item in ['cats','dogs']:
    dir = os.path.join(basedir, "test", item)
    image_files = glob.glob(os.path.join(dir,"*.jpg"))
    print("loding {}, image count={}".format(dir, len(image_files)))
    for image_file in image_files:
        image = cv2.imread(image_file)
        X_valid.append(cv2.resize(image, (128, 128)))
        label = np.zeros(2, dtype=np.uint8)
        label[cate_dict[item]]=1
        y_valid.append(label)
X_valid = np.array(X_valid)
y_valid = np.array(y_valid)

# 训练集和验证集大小
print(X_train.shape)
print(y_train.shape)
print(X_valid.shape)
print(y_valid.shape)

# 建立模型
base_model = VGG16(input_tensor=Input((128, 128, 3)), weights='imagenet', include_top=False)

# 设置VGG16模型中的层不参与训练
for layers in base_model.layers:
    layers.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.25)(x)
x = Dense(2, activation='softmax')(x)
model = Model(base_model.input, x)
model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=16, epochs=10, validation_data=(X_valid, y_valid))
model.save("models/vgg16-mymodel.h5")
