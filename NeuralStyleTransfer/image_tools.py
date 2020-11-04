# pjoshi, 11.2020
# python module containing tools for neural style transfer 

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import vgg19

def get_drive_images(COLABPATH):
    '''
    function to get path to content, style, and generated images in google drive
    '''
    base_image_path = COLABPATH + "/images/content.png"
    style_reference_image_path = COLABPATH + "/images/style.png"
    result_prefix = COLABPATH + "/output_images/generated"

    return (base_image_path, style_reference_image_path, result_prefix)

def get_test_images():
    '''
    function to get path to test content, style, and generated images
    '''
    base_image_path = keras.utils.get_file("paris.jpg", "https://i.imgur.com/F28w3Ac.jpg")
    style_reference_image_path = keras.utils.get_file(
        "starry_night.jpg", "https://i.imgur.com/9ooB60I.jpg"
    )
    result_prefix = "generated"
    return (base_image_path, style_reference_image_path, result_prefix)

def set_gen_img_dims(base_image_path=None,nrows=256):
    '''
    function to set the dimensions of the generated picture.
    '''
    width, height = keras.preprocessing.image.load_img(base_image_path).size
    img_nrows = 256
    img_ncols = int(width * img_nrows / height)

    return (img_nrows, img_ncols)



def preprocess_image(image_path, nrows=None, ncols=None):
    '''
    Util function to open, resize and format pictures into appropriate tensors
    '''
    img = keras.preprocessing.image.load_img(
        image_path, target_size=(nrows, ncols)
    )
    img = keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return tf.convert_to_tensor(img)


def deprocess_image(x, nrows=None, ncols=None):
    '''
    Util function to convert a tensor into a valid image
    '''
    x = x.reshape((nrows, ncols, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'    
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype("uint8")
    return x


def gram_matrix(x):
    '''
    The gram matrix of an image tensor (feature-wise outer product)
    '''
    x = tf.transpose(x, (2, 0, 1))
    features = tf.reshape(x, (tf.shape(x)[0], -1))
    gram = tf.matmul(features, tf.transpose(features))
    return gram


def style_loss(style, combination):
    '''
    The "style loss" is designed to maintain
    the style of the reference image in the generated image.
    It is based on the gram matrices (which capture style) of
    feature maps from the style reference image
    and from the generated image
    '''
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    # size = img_nrows * img_ncols
    size = 256 * 256
    return tf.reduce_sum(tf.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))



def content_loss(base, combination):
    '''
    An auxiliary loss function
    designed to maintain the "content" of the
    base image in the generated image
    '''
    return tf.reduce_sum(tf.square(combination - base))



def total_variation_loss(x, nrows=None, ncols=None):
    '''
    The 3rd loss function, total variation loss,
    designed to keep the generated image locally coherent
    '''
    a = tf.square(
        x[:, : nrows - 1, : ncols - 1, :] - x[:, 1:, : ncols - 1, :]
    )
    b = tf.square(
        x[:, : nrows - 1, : ncols - 1, :] - x[:, : nrows - 1, 1:, :]
    )
    return tf.reduce_sum(tf.pow(a + b, 1.25))


