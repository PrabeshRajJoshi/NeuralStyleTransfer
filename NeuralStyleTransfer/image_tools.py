# pjoshi, 11.2020
# python module containing tools for neural style transfer 

from tensorflow import keras

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