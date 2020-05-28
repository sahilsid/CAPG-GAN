import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as keras
# from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
import tensorflow_addons as tfa
from tensorflow_addons.layers import *

def build_generator():
    Ia     = Input((128,128,3),name='Ia')
    Pa     = Input((128,128,1),name='Pa')
    Pb     = Input((128,128,1),name='Pb')

    merge0= concatenate([Ia,Pa,Pb],axis=3)
    conv0 = Conv2D(64, 7, activation = 'relu', padding = 'same',strides=1, kernel_initializer = 'he_normal',name='conv0')(merge0)
    
    conv1 = Conv2D(64, 5, activation = 'relu', padding = 'same',strides=2, kernel_initializer = 'he_normal',name='conv1')(conv0)
    
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same',strides=2, kernel_initializer = 'he_normal',name='conv2')(conv1)
    
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same',strides=2, kernel_initializer = 'he_normal',name='conv3')(conv2)
    
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same',strides=2, kernel_initializer = 'he_normal',name='conv4')(conv3)
    
    flat  = Flatten()(conv4)
    fc1   = Dense(512,activation='relu',name='fc1')(flat)
    maxout= Maxout(256,name='maxout')(fc1)

    fc2   = Reshape((8,8,64),name='fc2')(Dense(4096,activation='relu')(maxout))

    dc0_1 = Conv2DTranspose(32,4,strides=4,padding='same',activation='relu',name='dc0_1')(fc2)
    
    dc0_2 = Conv2DTranspose(16,2,strides=2,padding='same',activation='relu',name='dc0_2')(dc0_1)
    
    dc0_3 = Conv2DTranspose(8,2,strides=2,padding='same',activation='relu',name='dc0_3')(dc0_2)
    
    merge1 = concatenate([fc2,conv4], axis = 3)
    dc1 = Conv2DTranspose(512,2,strides=2,padding='same',activation='relu',name='dc1')(merge1)
    
    merge2 = concatenate([dc1,conv3], axis = -1)
    dc2 = Conv2DTranspose(256,2,strides=2,padding='same',activation='relu',name='dc2')(merge2)
    
    merge3 = concatenate([dc2,conv2, AveragePooling2D(4)(Ia),dc0_1], axis = -1)
    dc3 = Conv2DTranspose(128,2,strides=2,padding='same',activation='relu',name='dc3')(merge3)
    
    merge4 = concatenate([dc3,conv1,AveragePooling2D(2)(Ia),dc0_2], axis = -1)
    dc4 = Conv2DTranspose(64,2,strides=2,padding='same',activation='relu',name='dc4')(merge4)
    
    conv5 = Conv2D(3, 3, activation = 'relu', padding = 'same',strides=1, kernel_initializer = 'he_normal',name='conv5')(dc2)
    
    conv6 = Conv2D(3, 3, activation = 'relu', padding = 'same',strides=1, kernel_initializer = 'he_normal',name='conv6')(dc3)

    merge5= concatenate([dc4,conv0,Ia,dc0_3], axis = -1)
    conv7 = Conv2D(64, 5, activation = 'relu', padding = 'same',strides=1, kernel_initializer = 'he_normal',name='conv7')(merge5)

    conv8 = Conv2D(32, 3, activation = 'relu', padding = 'same',strides=1, kernel_initializer = 'he_normal',name='conv8')(conv7)
    
    conv9 = Conv2D(3, 3, activation = 'relu', padding = 'same',strides=1, kernel_initializer = 'he_normal',name='conv9')(conv8)
    
    model = Model(inputs = [Ia,Pa,Pb], outputs = conv9)
    
    model.summary()
    
    return model
    
def build_discriminator(agent=1):
    Ia     = Input((128,128,3))
    Ib     = Input((128,128,3))
    Pb     = Input((128,128,1))

    if(agent==1):
      merge0 = concatenate([Ia,Ib],axis=3)
    elif(agent==2):
      merge0 = concatenate([Ib,Pb],axis=3)
    else:
      raise Exception("Invalid Agent Id")

    conv0_0 = Conv2D(64, 4, padding = 'same',strides=2, kernel_initializer = 'he_normal')(merge0)
    conv0_1 = InstanceNormalization(axis=3)(conv0_0)
    conv0 = LeakyReLU()(conv0_1)
    
    conv1_0 = Conv2D(128, 4, padding = 'same',strides=2, kernel_initializer = 'he_normal')(conv0)
    conv1_1 = InstanceNormalization(axis=3)(conv1_0)
    conv1 = LeakyReLU()(conv1_1)
    
    conv2_0 = Conv2D(256, 4, padding = 'same',strides=2, kernel_initializer = 'he_normal')(conv1)
    conv2_1 = InstanceNormalization(axis=3)(conv2_0)
    conv2 = LeakyReLU()(conv2_1)
    
    conv3_0 = Conv2D(512, 4, padding = 'same',strides=2, kernel_initializer = 'he_normal')(conv2)
    conv3_1 = InstanceNormalization(axis=3)(conv3_0)
    conv3 = LeakyReLU()(conv3_1)
    
    conv4_0 = Conv2D(512, 4,strides=1, padding='same',kernel_initializer = 'he_normal')(conv3)
    conv4_1 = InstanceNormalization(axis=3)(conv4_0)
    conv4 = LeakyReLU()(conv4_1)

    conv5 = Conv2D(1, 4,strides=1, padding='same',activation='sigmoid', kernel_initializer = 'he_normal')(conv4)

    if(agent==1):
      model = Model(inputs = [Ia,Ib], outputs = conv5)
    elif(agent==2):
      model = Model(inputs = [Ib,Pb], outputs = conv5)
    
    model.summary()
    
    return model