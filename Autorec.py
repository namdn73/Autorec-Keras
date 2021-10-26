from numpy.lib.function_base import append
import pandas as pd
import keras
from matplotlib import pyplot as plt
import numpy as np
from keras import optimizers, regularizers
from keras.layers import Input, Dense, Dropout
from keras.models import Model, load_model
from keras import backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import sklearn
import tensorflow as tf

class Autorec:
    def __init__(self):
        self.model = None
        
    def loss(self, y_true, y_pred):
        """
        Compute MSE loss 
        :param y_true: the true output
        :param y_pred: the predicted output
        :return: MSE loss 
        """
        mask = tf.math.divide_no_nan(tf.math.multiply(y_true, y_pred), tf.math.multiply(y_true, y_pred)) # mask includes 1 for observed rating position and 0 for unobserved rating position
        e = tf.math.subtract(y_true, y_pred)
        e = tf.math.multiply(e,mask) # ignore the missing value (having value of zero) in loss computation 
        se = tf.square(e)   
        mse = tf.reduce_sum(se)  
        return mse  # mean square error

    def RMSE(self, y_true, y_pred):
        """
        Compute root mean square error
        :param y_true: the true output
        :param y_pred: the predicted output
        :return: root mean square error
        """
        mask = tf.math.divide_no_nan(tf.math.multiply(y_true, y_pred), tf.math.multiply(y_true, y_pred)) 
        e = tf.math.subtract(y_true, y_pred)
        e = tf.math.multiply(e,mask)  # ignore the missing value (having value of zero) in loss computation 
        se = tf.square(e)  
        mse = tf.reduce_sum(se) / tf.math.count_nonzero(y_true, dtype='float32') 
        rmse = tf.math.sqrt(mse)
        return rmse  # root mean square error

    def Model(self, Xtrain, compression):
        self.Ntrain, self.D = Xtrain.shape

        inp = Input(shape=(self.D,))
        x1 = Dense(units=compression, activation='sigmoid', kernel_regularizer=regularizers.l2(1),
                        kernel_initializer=keras.initializers.truncated_normal(mean=0, stddev=0.03)) (inp)
        x2 = Dense(units=self.D, activation='linear', kernel_regularizer=regularizers.l2(1),
                        kernel_initializer=keras.initializers.truncated_normal(mean=0, stddev=0.03)) (x1)
        self.model = Model(inputs = inp, outputs = x2)
        print(self.model.summary())

        adam = optimizers.Adam(lr=1e-3)
        self.model.compile(optimizer=adam, loss=self.loss, metrics=[self.RMSE])

        return self

    def fit(self, Xtrain, y_val = None, batch_sz = 32, epoch = 10, save_best_path = None, save_all_path = None):
        callbacks_ = []
        if save_best_path is not None:
            save_best = ModelCheckpoint(save_best_path, monitor='RMSE', 
                                verbose=1, save_best_only=True, mode='min', period=1)
            callbacks_.append(save_best)
        if save_all_path is not None:
            save_all = ModelCheckpoint(save_all_path, verbose=1, period=5)
            callbacks_.append(save_all)

        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1,
                                    patience=10, verbose=1, mode="min", min_lr=0.00000001)
        callbacks_.append(reduce_lr)

        if y_val is not None:
            validation_data_ = (Xtrain, y_val)

        else:
            validation_data_ = None

        
        self.model.fit(x=Xtrain, y=Xtrain, batch_size=batch_sz, 
                                epochs=epoch,
                                # steps_per_epoch=int(np.ceil(self.Ntrain / batch_sz)),
                                validation_data = validation_data_, 
                                callbacks= callbacks_)
    
    def LoadModel(self, save_path):
        self.model = load_model(save_path, custom_objects={"loss": self.loss, "RMSE": self.RMSE})
        adam = optimizers.Adam(lr=1e-3)
        self.model.compile(optimizer=adam, loss=self.loss, metrics=[self.RMSE])

        return self