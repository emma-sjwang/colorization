# sjwang
# created on 2018/01/27

import numpy as np
from keras.models import Model
from keras import optimizers
from keras import regularizers
import os
import random
from PIL import Image
from keras.layers import Conv2D,UpSampling2D,concatenate,BatchNormalization
import cv2
import tensorflow as tf
from keras.layers import Input, Lambda
from keras import backend as K


ISOTIMEFORMAT = '%Y-%m-%d %X'
img_size = [224, 224]
num_classes = 2
mean_value = 120.0
output_size = [112, 112]
crop_size = [224, 224]


def generate_inputs_outputs(crop_image):
    '''
    seperate inputs and outputs
    :param crop_image:
    :return:
    '''
    crop_image = np.array(crop_image)
    crop_image = crop_image.astype(np.uint8)
    lab_image = cv2.cvtColor(crop_image, cv2.COLOR_RGB2LAB)
    input = lab_image[:,:,0]
    output = lab_image[:,:,1:3]
    return input, output


def get_patch(image_path):
    '''
    resize image from 256 to 224
    :param image_path:
    :return:
    '''
    image = Image.open(image_path)
    patch_image = image.resize((crop_size[0], crop_size[1]))
    return patch_image


def generate_image(val_data, val_label, predict_results, filename):
    '''
    save prediction results, generate contrast results. using opencv
    :param val_data:
    :param val_label:
    :param predict_results:
    :param filename:
    :return:
    '''
    prediction = np.zeros([val_data.shape[0], val_data.shape[1], 3], dtype=int)
    label = np.zeros([val_data.shape[0], val_data.shape[1], 3], dtype=int)

    val_data = val_data * mean_value + mean_value
    predict_results = predict_results[0,:,:,:]
    predict_results = predict_results * 255
    for x in range(val_data.shape[0]):
        for y in range(val_data.shape[1]):
            prediction[x,y,1:3] = predict_results[x/2,y/2,:]
    # save path
    save_path = "../data/results/baseline/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    prediction[:,:,0] = val_data
    label[:,:,0] = val_data
    label[:,:,1:3] = val_label

    grayscale = np.zeros([val_data.shape[0], val_data.shape[1],3])
    grayscale[:,:,0] = val_data
    grayscale[:,:,1] = val_data
    grayscale[:,:,2] = val_data

    img_pred = np.array(lab2rgb(prediction))
    img_label = np.array(lab2rgb(label))

    contrast_image = np.zeros([grayscale.shape[0], grayscale.shape[1]*3+6, grayscale.shape[2]], dtype=np.uint8)
    contrast_image[0:img_label.shape[0], 0:img_label.shape[1],:] = img_label
    contrast_image[0:grayscale.shape[0], img_label.shape[1] + 2:img_label.shape[1] + 2 + grayscale.shape[1],:] = grayscale
    contrast_image[0:img_pred.shape[0], img_label.shape[1] + 4 + grayscale.shape[1]:img_label.shape[1] + 4 + grayscale.shape[1]+img_pred.shape[1],:] = img_pred

    cv2.imwrite(save_path+filename[:-4]+".jpg", np.array(contrast_image))


def lab2rgb(im):
    im = im.astype(np.uint8)
    RGB = cv2.cvtColor(im, cv2.COLOR_LAB2BGR)
    return RGB


class myUnet(object):

    def __init__(self,q_train, q_val, img_rows = 224, img_cols = 224,channels = 1,n_class = 2, lr = 1e-2,
                 epochs = 100,train_iters = 41000, batch_size=16,every_epochs=1,val_num = 0, GPU_ID = 0,
                 filelist_path='',data_path_root='',model_name=''): #7056/6*8,
        # type: (object, object, object, object, object, object, object, object, object, object, object, object, object, object, object, object) -> object
        """

        :rtype: object
        """
        self.q_train = q_train
        self.q_val = q_val
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.epochs = epochs
        self.every_epochs = every_epochs
        self.train_iters = train_iters/batch_size+1
        self.batch_size = batch_size
        self.n_class = n_class
        self.channels = channels
        self.first_load = True
        self.val_num = val_num
        self.GPU_ID = GPU_ID
        self.lr = lr
        self.data_path_root = data_path_root
        self.filelist_path = filelist_path
        self.model_name = model_name

        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.GPU_ID)
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
        model = Model(inputs=inputs, outputs=output)

        adadelta = optimizers.Adadelta()
        model.compile(optimizer=adadelta, loss='mean_squared_error', metrics=['accuracy'])
        return model


    def test(self):

        model_name = self.model_name

        model = self.get_network()

        model.summary()
        # transfer multi-gpu model weights into single gpu weights.
        if not os.path.exists(model_name+"single"):
            models = _to_multi_gpu(model, [0,1,2,3])
            if os.path.exists(model_name):
                print "Loading model from %s" % model_name
                models.load_weights(model_name)
            else:
                print "ERROR: NO weights file %s." % model_name
                exit(-1)
            model.save_weights(model_name+"single")

        model_name = model_name+"single"
        model = self.get_network()
        if os.path.exists(model_name):
            print "Loading model from %s" % model_name
            model.load_weights(model_name)
        else:
            print "ERROR: NO weights file %s." % model_name
            exit(-1)

        # data root path
        data_path_root = self.data_path_root
        # test data filename list
        filelist_path = self.filelist_path
        if not os.path.exists(filelist_path):
            print "ERROR: NO filelist txt file %s." % filelist_path
            exit(-1)
        filelist_file = open(filelist_path)
        filename_list = filelist_file.read().splitlines()
        random.shuffle(filename_list)
        filelist_file.close()
        index = 0
        for filename in filename_list:
            index += 1
            print filename
            image_path = os.path.join(data_path_root, filename)
            if os.path.exists(image_path):
                crop_image = get_patch(image_path)
                temp_image = np.array(crop_image)
                # if not RGB image drop it.
                if len(temp_image.shape) < 3:
                    continue
                inputs, outputs = generate_inputs_outputs(crop_image)
                val_data = (inputs - mean_value) / mean_value
                val_label = outputs

                _val_data = np.zeros([1,val_data.shape[0], val_data.shape[1], 1])
                _val_data[0,:,:,0] = val_data
                predict_results = model.predict(_val_data, batch_size=1, verbose=1)
                # visualization
                generate_image(val_data, val_label, predict_results, filename)


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


def consume(q_train, q_val, GPU_ID, filelist_path, data_path_root, model_name):
    myunet = myUnet(q_train, q_val, lr = 1e-3, batch_size=128, every_epochs=1, GPU_ID = GPU_ID,
                    filelist_path=filelist_path,data_path_root=data_path_root,model_name=model_name)
    myunet.test()


if __name__ == '__main__':
    # GPU_ID = int(sys.argv[1])

    GPU_ID = 0
    filelist_path = "../data/test_data_list.txt"
    model_name = "../model/0119_nomean_weights.599-14717.3408.hdf5"
    data_path_root = '../data/testSetPlaces205_resize/data/'
    consume(1, 1, GPU_ID=GPU_ID, filelist_path=filelist_path,data_path_root=data_path_root,model_name=model_name)