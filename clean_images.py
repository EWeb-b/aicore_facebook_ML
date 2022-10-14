import os
import pandas as pd
import numpy as np
from PIL import Image

def clean_image_data(path:str) -> None:
    """Takes in a filepath to a folder of images, cleans them, then saves them in 'cleaned_images' folder.

    Args:
        path: The path to the folder of images to be cleaned.
    """
    dirs = os.listdir(path)
    final_size = 512

    for item in dirs:
        im = Image.open(path + item)
        new_im = resize_image(final_size, im)
        new_im.save(f"data/cleaned_images/{item}")

def resize_image(final_size: int, im: Image) -> Image:
    """Resizes the image and saves it as an RGB type image.

    Args:
        final_size: The new size of the horizontal and vertical axes of the image, in pixels.
        im: The image to be resized.

    Returns:
        new_im: The new, resized image.
    """
    size = im.size
    ratio = float(final_size) / max(size)
    new_image_size = tuple([int(x*ratio) for x in size])
    im = im.resize(new_image_size, Image.ANTIALIAS)
    new_im = Image.new("RGB", (final_size, final_size))
    new_im.paste(im, ((final_size-new_image_size[0])//2, (final_size-new_image_size[1])//2))
    return new_im

def check_images() -> None:
    """Checks the images in the cleaned_images folder and prints to the terminal if they don't have 3 channels and/or not RGB.
    """
    dirs = os.listdir("data/cleaned_images")
    for item in dirs:
        im = Image.open("data/cleaned_images/" + item)
        im_size = im.size
        im_arr = np.asarray(im)
        if im_arr.shape[2] != 3:
            print(item)
            print(im_arr.shape)
            print("\n")
        if im.mode != "RGB":
            print(item)
            print(im.mode)
            print("\n")
        if im_size != (512,512):
            print("Wrong size: ", im_size)

if __name__ == '__main__':
    path = "data/cleaned_images"
    clean_image_data("data/images")
    # check_images()





            








