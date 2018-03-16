# sjwang
# created on 2018/01/23

import numpy as np
import os
import cv2
import random
from PIL import Image


CPU_Kernel = 8
output_size = [112, 112]
crop_size = [224, 224]


def produce_data(q, phase, data_path_root, filelist):
    if phase == 'train':
        data_generator(q, phase, data_path_root, filelist)
    elif phase == 'val':
        data_generator(q, phase, data_path_root, filelist)


def data_generator(q, phase, data_path_root, filelist):
    '''
    generate data and label, store them in queue q.
    :param q: queue
    :param phase:  train or val
    :param data_path_root:
    :param filelist:
    :return:
    '''
    batch_size = 1
    batch_image = np.zeros([batch_size,crop_size[0],crop_size[1]])
    batch_label = np.zeros([batch_size,output_size[0],output_size[1], 2])

    filename_list_file = open(filelist)
    filename_list = filename_list_file.read().splitlines()
    random.shuffle(filename_list)
    filename_list_file.close()
    while (len(filename_list) > batch_size):
        i = 0
        mean = 0
        # start = time.time()
        while i < batch_size:
            # random get image name
            rand = np.random.randint(0, len(filename_list))
            imagename = filename_list[rand]

            del filename_list[rand]
            image_path = os.path.join(data_path_root, imagename)

            if os.path.exists(image_path):
                crop_image = get_patch(image_path)
                temp_image = np.array(crop_image)

                # if the input is grey image, drop it
                if len(temp_image.shape) < 3:
                    continue
                temp_mean = np.mean(np.mean(temp_image, axis=0), axis=0)
                a = np.fabs(temp_mean[0]-temp_mean[1])
                b = np.fabs(temp_mean[0]-temp_mean[2])

                # if the image with similar RGB mean, then drop it
                if a < 1 and b < 1:
                    continue

                inputs, outputs = generate_inputs_outputs(crop_image)
                mean = mean + np.mean(inputs)
                resized_output = np.zeros([output_size[0], output_size[1], 2])

                # resize to 112 for labels
                for x in range(0, outputs.shape[0], 2):
                    for y in range(0, outputs.shape[1], 2):
                        resized_output[x/2,y/2,:] = outputs[x,y,:]
                # normalization for input
                resized_output = np.array(resized_output) / 255.0

                batch_image[i, :, :] = inputs
                batch_label[i, :, :, :] = resized_output

            else:
                print "ERROR: CANNOT find image in %s" % image_path
                continue
            i = i + 1
        mean = mean / batch_size * 1.0
        q.put([batch_image, batch_label, mean])
    return


def generate_inputs_outputs(crop_image):
    '''
    generator grey image and AB color space image
    :param crop_image: PIL object
    :return: numpy array image
    '''
    augmentation_image = data_augmentation(crop_image)
    if augmentation_image.mode != "RGB":
        augmentation_image = augmentation_image.convert("RGB")
    lab_image = cv2.cvtColor(np.array(augmentation_image).astype(np.uint8), cv2.COLOR_RGB2LAB)
    input = lab_image[:,:,0]
    output = lab_image[:,:,1:3]
    return input, output


def get_patch(image_path):
    '''
    random crop image from 256*256 to 224*224
    :param image_path:
    :return: PIL object image
    '''
    image = Image.open(image_path)
    shape = image.size
    x_range = shape[0] - crop_size[0]
    y_range = shape[1] - crop_size[1]
    x_begin = random.randint(0, x_range)
    y_begin = random.randint(0, y_range)
    x_end = x_begin + crop_size[0]
    y_end = y_begin + crop_size[1]
    patch_image = image.crop((y_begin, x_begin, y_end, x_end))

    return patch_image


def  data_augmentation(crop_image):
    '''
    random flip and rotation
    :param crop_image: a PIL object
    :return:
    '''
    # flip
    rand_flip = random.random()
    if rand_flip > 0.67:
        crop_image = crop_image.transpose(Image.FLIP_LEFT_RIGHT)
    elif rand_flip > 0.33:
        crop_image = crop_image.transpose(Image.FLIP_TOP_BOTTOM)

    # rotation
    # Image.ROTATE_90:2
    # Image.ROTATE_180:3
    # Image.ROTATE_270:4
    j = random.randint(2,5)
    rotation_image = crop_image.transpose(j)

    return rotation_image
