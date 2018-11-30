"""
The U-Net scaled for 128x128 inputs
"""
from keras.layers import Input
from keras.layers.core import Activation
from keras.layers.convolutional import Conv2D, Conv3D
from keras.layers import Concatenate, Reshape, Add, MaxPooling2D,Flatten, Dense, Lambda,MaxPooling3D,Conv3DTranspose
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers import UpSampling2D, UpSampling3D, Dropout, Multiply
from keras import regularizers
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from keras.losses import mean_squared_error

from keras import backend as K


def IoU_loss(y_true,y_pred):
    smooth = 1e-12
    # author = Vladimir Iglovikov
    intersection = K.sum(y_true * y_pred)#,axis=[1,2,3,4])
    sum_ = K.sum(y_true + y_pred)#,axis=[1,2,3,4])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(1-jac)


def IoU(y_true,y_pred):
    # author = Vladimir Iglovikov
    smooth = 1e-12
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))

    intersection = K.sum(y_true * y_pred_pos)#,axis=[1,2,3,4])
    sum_ = K.sum(y_true + y_pred)#,axis=[1,2,3,4])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)    

def attraction_loss(y_true,y_pred):
    smooth = 1e-12
    intersection = K.sum(y_true * y_pred)
    sum_ = K.sum(y_true)
    value = (intersection+smooth)/(sum_+smooth)
   
    
    
    return K.mean(1-value)# + K.mean(Lp)

    
    


class unet3D():

    def __init__(self, sz, bot_neck, nf, regul, dropout,seed, bn=True):

        #bot_neck is the size for which the bottle neck should be one
        #sz is the side of the cube
        #bn - if true, add batchnorm
        
        
        self.sz = sz
        #self.nlayers = int(np.floor(np.log(sz)/np.log(2)))+1
        self.nlayers = int(np.floor(np.log(bot_neck)/np.log(2)))+1
        self.nf = nf
        self. regul = regul
        self.dropout = dropout
        self.seed = seed
        self.bn = bn
        
    def createModel(self):


        nf = self.nf
        regul = self.regul
        dropout = self.dropout
        seed = self.seed
        sz = self.sz
    
        inputs = Input((self.sz, self.sz,self.sz,1))
#        points = Input((self.sz, self.sz,self.sz,1))

        conv1 = []


        conv = Conv3D(nf, (3, 3, 3), padding='same')(inputs)
#        conv = Concatenate(axis=-1)([conv,points])
        if self.bn:
            conv = BatchNormalization(axis=-1)(conv)
        conv = Activation('relu')(conv)

        conv1.append(conv)
        c = -1
        for ii in range(1, self.nlayers):
            c+=1
            if ii < 8:
                nnf = nf * ii
            else:
                nnf = nnf * 8
            if ii == self.nlayers-1:
                conv = Conv3D(nnf, (2, 2, 2), padding='same', strides=(2, 2, 2),kernel_regularizer=regularizers.l2(regul))(conv1[ii-1]) #last layer
            else:

                    conv = Conv3D(nnf, (2, 2, 2), padding='same', strides=(2, 2, 2),kernel_regularizer=regularizers.l2(regul))(conv1[ii-1])
                    
            if dropout !=0:
                conv = Dropout(dropout,seed = seed)(conv)
            if self.bn:
                conv = BatchNormalization(axis=-1)(conv)
            conv1.append( Activation('relu')(conv) )
        upconv1 = conv1[-1]
        ind = 0
        c=-1
        for xx in range(self.nlayers-2, -1, -1):
            c+=1
            if xx < 8 and xx > 0:
                nnf = nf * xx
            elif xx <= 0:
                nnf = 1
            else:
                nnf = nnf * 8
            print(xx)
            #up = merge([UpSampling2D(size=(2, 2))(upconv1), conv1[xx]], mode='concat', concat_axis=1)
            up = Concatenate(axis=-1)([UpSampling3D(size=(2, 2, 2))(upconv1), conv1[xx]])
            upconv = Conv3D(nnf * 8, (3, 3, 3), padding="same", kernel_regularizer=regularizers.l2(regul))(up)
            if self.bn:
                upconv = BatchNormalization(axis=-1)(upconv)
            upconv = Activation('relu')(upconv)
            # conv7 = Dropout(0.2)(conv7)
            if xx > 0:
                upconv = Conv3D(nnf * 8, (3, 3, 3), padding="same", kernel_regularizer=regularizers.l2(regul))(upconv)
                #if dropout !=0:
                #    conv = Dropout(dropout,seed = seed)(upconv) # TO DO - change (25JUl)
                if self.bn:
                    upconv = BatchNormalization(axis=-1)(upconv)
                upconv1 = Activation('relu')(upconv)
                ind += 1
                # nf*8 x 2 x 2
            else:
                pass

#        upconv = Concatenate(axis=-1)([upconv,points])

        upconv = Conv3D(1, (3, 3, 3), padding='same')(upconv)
        
        

        out = Activation('sigmoid',name='segmentation')(upconv)
        
        print(K.int_shape(out))
        


        model = Model(inputs=inputs, outputs=out)

        return model


    def createGuidedModel(self):


        nf = self.nf
        regul = self.regul
        dropout = self.dropout
        seed = self.seed
        sz = self.sz
    
        inputs = Input((self.sz, self.sz,self.sz,1))
        points = Input((self.sz, self.sz,self.sz,1))
        segmentation = Input((self.sz, self.sz,self.sz,1))

        conv1 = []


        conv = Conv3D(nf, (3, 3, 3), padding='same')(inputs)
        conv = Concatenate(axis=-1)([conv,points,segmentation])
        if self.bn:
            conv = BatchNormalization(axis=-1)(conv)
        conv = Activation('relu')(conv)

        conv1.append(conv)
        c = -1
        for ii in range(1, self.nlayers):
            c+=1
            if ii < 8:
                nnf = nf * ii
            else:
                nnf = nnf * 8
            if ii == self.nlayers-1:
                conv = Conv3D(nnf, (2, 2, 2), padding='same', strides=(2, 2, 2),kernel_regularizer=regularizers.l2(regul))(conv1[ii-1]) #last layer
            else:

                    conv = Conv3D(nnf, (2, 2, 2), padding='same', strides=(2, 2, 2),kernel_regularizer=regularizers.l2(regul))(conv1[ii-1])
                    
            if dropout !=0:
                conv = Dropout(dropout,seed = seed)(conv)
            if self.bn:
                conv = BatchNormalization(axis=-1)(conv)
            conv1.append( Activation('relu')(conv) )
        upconv1 = conv1[-1]
        ind = 0
        c=-1
        for xx in range(self.nlayers-2, -1, -1):
            c+=1
            if xx < 8 and xx > 0:
                nnf = nf * xx
            elif xx <= 0:
                nnf = 1
            else:
                nnf = nnf * 8
            print(xx)
            #up = merge([UpSampling2D(size=(2, 2))(upconv1), conv1[xx]], mode='concat', concat_axis=1)
            up = Concatenate(axis=-1)([UpSampling3D(size=(2, 2, 2))(upconv1), conv1[xx]])
            upconv = Conv3D(nnf * 8, (3, 3, 3), padding="same", kernel_regularizer=regularizers.l2(regul))(up)
            if self.bn:
                upconv = BatchNormalization(axis=-1)(upconv)
            upconv = Activation('relu')(upconv)
            # conv7 = Dropout(0.2)(conv7)
            if xx > 0:
                upconv = Conv3D(nnf * 8, (3, 3, 3), padding="same", kernel_regularizer=regularizers.l2(regul))(upconv)
                #if dropout !=0:
                #    conv = Dropout(dropout,seed = seed)(upconv) # TO DO - change (25JUl)
                if self.bn:
                    upconv = BatchNormalization(axis=-1)(upconv)
                upconv1 = Activation('relu')(upconv)
                ind += 1
                # nf*8 x 2 x 2
            else:
                pass

#        upconv = Concatenate(axis=-1)([upconv,points])

        upconv = Conv3D(1, (3, 3, 3), padding='same')(upconv)
        
        

        out = Activation('sigmoid',name='segmentation')(upconv)
        
        print(K.int_shape(out))
        


        model = Model(inputs=[inputs,points,segmentation], outputs=out)

        return model

        
    def correctionModel(self, weights):
        volume = Input((self.sz, self.sz,self.sz,1))
        mapping = Input((self.sz, self.sz,self.sz,1))        
#        segmentation = Input((self.sz, self.sz,self.sz,1))

        seg_model = self.createModel()
        seg_model.load_weights(weights)
        seg_model.trainable=False
    
        segmentation = seg_model(volume)
        
        correction_model = self.createGuidedModel()
        seg = correction_model([volume,mapping,segmentation])
        seg = Activation('linear',name='segmentation')(seg)
        aux = Activation('linear',name='attraction')(seg)
        init = Activation('linear',name='initial')(segmentation)
        
        model = Model(inputs=[volume,mapping],outputs=[seg,aux,init])
        return model
        
    def log(self, hist, params=False, file = 'log.txt'):

        if params is True:
            out = open(file, 'ab+')
            out.write((hist.params + '\n'))
            out.write((hist.epoch + hist.history + '\n'))
            out.close()
        else:
            out = open(file, 'ab+')
            out.write((hist.epoch + hist.history + '\n'))
            out.close()