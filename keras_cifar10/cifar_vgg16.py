# coding: utf-8
"""
Create on 2018/11/1

@author:hexiaosong
"""
from keras.models import Model
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.applications import VGG16
from keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input

batch_size = 16
epochs = 10
num_classes = 10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# 建立模型
base_model = VGG16(input_tensor=Input(x_train.shape[1:]), weights='imagenet', include_top=False)

# 设置VGG16模型中的层不参与训练
for layers in base_model.layers:
    layers.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.25)(x)
x = Dense(num_classes, activation='softmax')(x)
model = Model(base_model.input, x)
model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))
model.save("models/vgg16-mymodel.h5")
