# sjwang
# created on 2018/01/27

import numpy as np
import cv2
import os
import tensorflow as tf

from keras.models import Model
from keras.layers import Conv2D,UpSampling2D,concatenate,BatchNormalization, Input, Lambda
from keras import optimizers
from keras.optimizers import *
from keras import backend as K
from keras import regularizers
from keras.layers.merge import Concatenate
from keras.callbacks import TensorBoard, LearningRateScheduler, ModelCheckpoint

# you can change parameters here

ISOTIMEFORMAT = '%Y-%m-%d %X'
img_size = [224, 224]
num_classes = 2
mean_value = 120.0

logs_path = "../log_tf/baseline/0204/"
# if you want to finetuning, changge the below path to your model path
Load_weights_from = "../model/0119_nomean_weights.599-14717.3408.hdf5single"
# save model weights to where
save_weights_to = "../model/0120_finetuning_nomean_weights_adam_amsgrad_{epoch:03d}_{loss:.1f}.hdf5"
save_weights_to = "../model/0204.hdf5"


def mse(y_true, y_pred):
    y_pred = K.reshape(y_pred, [-1, 2])
    y_true = K.reshape(y_true, [-1, 2])
    loss = K.sum(K.square(y_pred - y_true))
    return loss


def load_train_data(q, batch_size = 8):
    while 1:
        each_queue = 1
        n = batch_size / each_queue
        mean = 0
        i = 0
        label_batch = np.zeros([batch_size, img_size[0]/2, img_size[1]/2, 2])
        data_batch = np.zeros([batch_size, img_size[0], img_size[1],1])
        while i < n:
            [data, label, _mean ] = q.get()
            mean = mean + _mean
            data = data.astype('float32')
            data = data - mean_value
            data_batch[i*each_queue:(i+1)*each_queue, :,:,0] = data
            label_batch[i*each_queue:(i+1)*each_queue, :,:,:] = label
            i = i + 1

        data_batch = data_batch.astype('float32')
        label_batch = label_batch.astype('float32')
        yield data_batch / mean_value, label_batch


def transfer_label(int_labels, num_classes):
    shape = int_labels.shape
    categorical_labels = np.zeros([shape[0], shape[1], shape[2], num_classes])
    categorical_labels[:, :, :, 1] = int_labels[:, :, :, 0]
    categorical_labels[:, :, :, 0] = 1 - int_labels[:, :, :, 0]
    return categorical_labels


def load_val_data(q):
    [data_batch, label_batch, rate] = q.get()
    data_batch = data_batch.astype('float32')
    data_batch = data_batch - mean_value
    label_batch = transfer_label(label_batch, num_classes)
    label_batch = label_batch.astype('float32')
    return data_batch / 240, label_batch, rate


class myUnet(object):

    def __init__(self,q_train, q_val, img_rows = 224, img_cols = 224,channels = 1,n_class = 2, lr = 1e-2,
                 epochs = 1000,train_iters = 41000, batch_size=144,batch_num=2,every_epochs=1,val_num = 0, GPU_ID = [0],
                 data_path_root=''): #7056/6*8,
        self.q_train = q_train
        self.q_val = q_val
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.epochs = epochs
        self.every_epochs = every_epochs
        self.batch_size = batch_size
        self.n_class = n_class
        self.channels = channels
        self.first_load = True
        self.val_num = val_num
        self.GPU_ID = GPU_ID
        self.batch_num = batch_num
        self.train_iters = train_iters / (batch_size * batch_num*len(GPU_ID))+ 1
        self.lr = lr
        self.data_path_root = data_path_root
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.GPU_ID).strip('[').strip(']').replace(", ", ",")
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.9
        session = tf.Session(config=config)


    def get_network(self):
        penalty = 0.001
        inputs = Input((self.img_rows, self.img_cols, 1))

        conv1_1 = Conv2D(64, (3,3), activation = 'relu', strides=(2, 2), padding = 'same', kernel_initializer = 'glorot_normal',kernel_regularizer=regularizers.l2(penalty),name = 'conv1_1')(inputs)
        bn1_1 = BatchNormalization()(conv1_1)
        conv1_2 = Conv2D(128, (3,3), activation = 'relu', strides=(1, 1), padding = 'same', kernel_initializer = 'glorot_normal',kernel_regularizer=regularizers.l2(penalty),name = 'conv1_2')(bn1_1)
        bn1_2 = BatchNormalization()(conv1_2)

        conv2_1 = Conv2D(128, (3,3), activation = 'relu', strides=(2, 2), padding = 'same', kernel_initializer = 'glorot_normal',kernel_regularizer=regularizers.l2(penalty),name = 'conv2_1')(bn1_2)
        bn2_1 = BatchNormalization()(conv2_1)
        conv2_2 = Conv2D(256, (3,3), activation = 'relu', strides=(1, 1), padding = 'same', kernel_initializer = 'glorot_normal',kernel_regularizer=regularizers.l2(penalty),name = 'conv2_2')(bn2_1)
        bn2_2 = BatchNormalization()(conv2_2)

        conv3_1 = Conv2D(256, (3,3), activation = 'relu', strides=(2, 2), padding = 'same', kernel_initializer = 'glorot_normal',kernel_regularizer=regularizers.l2(penalty),name = 'conv3_1')(bn2_2)
        bn3_1 = BatchNormalization()(conv3_1)
        conv3_2 = Conv2D(512, (3,3), activation = 'relu', strides=(1, 1), padding = 'same', kernel_initializer = 'glorot_normal',kernel_regularizer=regularizers.l2(penalty),name = 'conv3_2')(bn3_1)
        bn3_2 = BatchNormalization()(conv3_2)

        conv4_1 = Conv2D(512, (3,3), activation = 'relu', strides=(1, 1), padding = 'same', kernel_initializer = 'glorot_normal',kernel_regularizer=regularizers.l2(penalty),name = 'conv4_1')(bn3_2)
        bn4_1 = BatchNormalization()(conv4_1)
        conv4_2 = Conv2D(256, (3,3), activation = 'relu', strides=(1, 1), padding = 'same', kernel_initializer = 'glorot_normal',kernel_regularizer=regularizers.l2(penalty),name = 'conv4_2')(bn4_1)
        bn4_2 = BatchNormalization()(conv4_2)

        conv5_1 = Conv2D(128, (3,3), activation = 'relu', strides=(1, 1), padding = 'same', kernel_initializer = 'glorot_normal',kernel_regularizer=regularizers.l2(penalty),name = 'conv5_1')(bn4_2)
        bn5_1 = BatchNormalization()(conv5_1)

        upsample_1 = UpSampling2D(size=(2, 2))(bn5_1)
        conv6_1 = Conv2D(64, (3,3), activation = 'relu', strides=(1, 1), padding = 'same', kernel_initializer = 'glorot_normal',kernel_regularizer=regularizers.l2(penalty),name = 'conv6_1')(upsample_1)
        bn6_1 = BatchNormalization()(conv6_1)
        conv6_2 = Conv2D(64, (3,3), activation = 'relu', strides=(1, 1), padding = 'same', kernel_initializer = 'glorot_normal',kernel_regularizer=regularizers.l2(penalty),name = 'conv6_2')(bn6_1)
        bn6_2 = BatchNormalization()(conv6_2)


        upsample_2 = UpSampling2D(size=(2, 2))(bn6_2)
        conv7_1 = Conv2D(32, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'glorot_normal',kernel_regularizer=regularizers.l2(penalty),name = 'conv7_1')(upsample_2)
        bn7_1 = BatchNormalization()(conv7_1)
        output = Conv2D(2, (3,3), activation = 'sigmoid', padding = 'same', kernel_initializer = 'glorot_normal',kernel_regularizer=regularizers.l2(penalty),name = 'conv7_2')(bn7_1)

        self.model = Model(inputs=inputs, outputs=output)

        if os.path.exists(Load_weights_from):
            print "Loading model from %s" % Load_weights_from
            self.model.load_weights(Load_weights_from)

        models = _to_multi_gpu(self.model, self.GPU_ID)
        adam = optimizers.Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.001)
        models.compile(optimizer=adam, loss=mse, metrics=['accuracy'])

        self.models = models
        return models


    def scheduler(self, epoch):
        # save model every 10 epoch
        if epoch % 10 == 0:
            K.set_value(self.models.optimizer.lr, 0.9*(K.get_value(self.models.optimizer.lr)))
        return K.get_value(self.models.optimizer.lr)


    def train(self):

        print("----------------------Begin creating unet network-------------------------")
        models = self.get_network()
        models.summary()

        logswriter = tf.summary.FileWriter
        print "logtf path %s " % logs_path
        if not os.path.exists(logs_path):
            os.makedirs(logs_path)
        self.summary_writer = logswriter(logs_path)

        board = TensorBoard(log_dir=logs_path, batch_size=self.batch_size)
        checkpointer = ModelCheckpoint(filepath=save_weights_to, monitor='loss', verbose=1, save_weights_only=True)
        change_lr = LearningRateScheduler(self.scheduler)

        history = models.fit_generator(
            generator=load_train_data(self.q_train, self.batch_size * self.batch_num * len(self.GPU_ID)),
            steps_per_epoch=self.train_iters, epochs=self.epochs, verbose=1,
         callbacks=[board, checkpointer, change_lr])



def _slice_batch( x, n_gpus, part):
    sh = K.shape(x)
    L = sh[0] / n_gpus
    if part == n_gpus - 1:
        return x[part * L:]
    return x[part * L:(part + 1) * L]

def _to_multi_gpu( model, gpuList=[0]):

    with tf.device('/cpu:0'):
        x = Input(model.input_shape[1:])
    towers = []
    for g in xrange(len(gpuList)):
        with tf.device('/gpu:' + str(g)):
            slice_g = Lambda(_slice_batch, lambda shape: shape, arguments={'n_gpus': len(gpuList), 'part': g})(
                x)
            towers.append(model(slice_g))
    with tf.device('/cpu:0'):
        if len(gpuList) == 1:
            merged = towers[0]
        else:
            merged = concatenate(towers, axis=0)
    return Model(inputs=[x], outputs=merged)


def slice_batch(x, n_gpus, part):
    sh = K.shape(x)
    L = sh[0] // n_gpus
    if part == n_gpus - 1:
        return x[part * L:]
    return x[part * L:(part + 1) * L]


def to_multi_gpu(model, n_gpus=2):
    if n_gpus == 1:
        return model

    with tf.device('/cpu:0'):
        x = Input(model.input_shape[1:])
    towers = []
    for g in range(n_gpus):
        with tf.device('/gpu:' + str(g + 1)):
            slice_g = Lambda(slice_batch, lambda shape: shape, arguments={'n_gpus': n_gpus, 'part': g})(x)
            towers.append(model(slice_g))

    with tf.device('/cpu:0'):
        merged = Concatenate(axis=0)(towers)
        return Model(inputs=[x], outputs=merged)


def consume(q_train, q_val, GPU_ID, data_path_root):
    myunet = myUnet(q_train, q_val, lr = 1e-7, batch_size=64, batch_num=1, every_epochs=1,epochs = 600, GPU_ID = GPU_ID, data_path_root=data_path_root)
    myunet.train()
