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
    The "style loss" is designed to maintain the style of the reference image 
    in the generated image. It is based on the gram matrices (which capture style) 
    of feature maps from the style reference image and from the generated image
    '''
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    # size = img_nrows * img_ncols
    size = 256 * 256
    return tf.reduce_sum(tf.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))



def content_loss(base, combination):
    '''
    An auxiliary loss function designed to maintain the "content" of the
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


class StyleTransferLoss:
    def __init__(self):
        # set the names of layers that will be used for loss computations
        self.style_layer_names = [
                                "block1_conv1",
                                "block2_conv1",
                                "block3_conv1",
                                "block4_conv1",
                                "block5_conv1",
                                ]
        self.content_layer_name = "block5_conv2"

        # Set the weights of the different loss components
        self.total_variation_weight = 1e-6
        self.style_weight = 1e-6
        self.content_weight = 2.5e-8

    def setup_feature_extractor(self):
        '''
        Method to setup the model that retrieves the intermediate activations of VGG19 (as a dict, by name).
        '''
        # Build a VGG19 model loaded with pre-trained ImageNet weights
        model = vgg19.VGG19(weights="imagenet", include_top=False)

        # Get the symbolic outputs of each "key" layer (we gave them unique names).
        self.outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

        # Set up a model that returns the activation values for every layer in
        # VGG19 (as a dict).
        self.feature_extractor = keras.Model(inputs=model.inputs, outputs=self.outputs_dict)
    

    def get_layer_names(self):
        '''
        Method to retrieve the names of model layers as a list.
        This helps decide which layers to choose for loss computation.
        '''
        if self.outputs_dict:
            return list(self.outputs_dict.keys())
        else:
            print("Use setup_feature_extractor first!")
    
    def set_style_layer_names(self):
        '''
        Method to set the list of layer-names (one or more) used to compute style loss
        '''
        pass
    
    def set_content_layer_name(self):
        '''
        Method to set the name of the layer (only one) used to compute content loss
        '''
        pass
    

    def compute_loss(self):
        '''
        Method to compute the total loss during style transfer to base(content) image.
        '''
        input_tensor = tf.concat(
            [self.base_image, self.style_image, self.combination_image], axis=0
        )
        features = self.feature_extractor(input_tensor)

        # Initialize the loss
        loss = tf.zeros(shape=())

        # Add content loss
        layer_features = features[self.content_layer_name]
        base_image_features = layer_features[0, :, :, :]
        combination_features = layer_features[2, :, :, :]
        loss = loss + self.content_weight * content_loss(
            base_image_features, combination_features
        )
        # Add style loss
        for layer_name in self.style_layer_names:
            layer_features = features[layer_name]
            style_features = layer_features[1, :, :, :]
            combination_features = layer_features[2, :, :, :]
            sl = style_loss(style_features, combination_features)
            loss += (self.style_weight / len(self.style_layer_names)) * sl

        # Add total variation loss
        loss += self.total_variation_weight * total_variation_loss(self.combination_image, nrows=self.base_image.shape[1], ncols=self.base_image.shape[2] )
        return loss
    
    @tf.function
    def compute_loss_and_grads(self, base_image=None, style_image=None, combination_image=None):
        '''
        Method with a tf.function decorator for loss & gradient computation
        THe decorator is used to compile it, and thus make it fast.
        '''
        self.base_image = base_image
        self.style_image = style_image
        self.combination_image = combination_image
        with tf.GradientTape() as tape:
            loss = self.compute_loss()
        grads = tape.gradient(loss, combination_image)
        return loss, grads

