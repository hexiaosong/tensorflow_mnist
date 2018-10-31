# coding: utf-8
"""
Create on 2018/10/31

@author:hexiaosong
"""
from keras.applications import VGG16
from keras import models,layers
from keras.callbacks import TensorBoard

from keras.preprocessing.image import ImageDataGenerator

train_dir="/Users/apple/Downloads/catvsdog/train"
test_dir="/Users/apple/Downloads/catvsdog/test"

train_pic_gen=ImageDataGenerator(rescale=1./255,rotation_range=20,width_shift_range=0.2,height_shift_range=0.2,
                                 shear_range=0.2,zoom_range=0.5,horizontal_flip=True,fill_mode='nearest')
test_pic_gen=ImageDataGenerator(rescale=1./255)

train_flow=train_pic_gen.flow_from_directory(train_dir,(128,128),batch_size=32,class_mode='binary')
test_flow=test_pic_gen.flow_from_directory(test_dir,(128,128),batch_size=32,class_mode='binary')

conv_base=VGG16(weights='imagenet',include_top=False,input_shape=(128,128,3))

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

conv_base.trainable=False

model.summary()

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])


model.fit_generator(
      train_flow,
      steps_per_epoch=32,
      epochs=50,
      validation_data=test_flow,
      validation_steps=32,callbacks=[TensorBoard(log_dir='/Users/apple/Downloads/catvsdog/logs/3')])
model.save_weights('outputs/weights_vgg16_use.h5')
