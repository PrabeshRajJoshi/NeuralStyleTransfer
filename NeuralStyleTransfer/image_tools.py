# pjoshi, 11.2020
# python module containing tools for neural style transfer 

def get_drive_images(COLABPATH):
    '''
    function to retrieve path to content, style, and generated images
    '''
    base_image_path = COLABPATH + "/images/content.png"
    style_reference_image_path = COLABPATH + "/images/style.png"
    result_prefix = COLABPATH + "/output_images/generated"

    return (base_image_path, style_reference_image_path, result_prefix)