import os
import pandas as pd
import numpy as np
from PIL import Image
from PIL import ImageOps

def clean_image_data(final_size: int, path:str) -> None:
    """Resizes and ensures images are of type RGB, then saves them in 'cleaned_images' folder.

    Args:
        final_size: The width and height of the new image, in pixels.
        path: The path to the folder of images to be cleaned.
    """
    dirs = os.listdir(path)

    for item in dirs:
        im = Image.open(path + item)
        new_im = ImageOps.pad(im, (final_size, final_size))
        new_im = new_im.convert('RGB')
        new_im.save(f"data/cleaned_images/{item}")

def check_images(final_size: int) -> None:
    """Checks the images in the cleaned_images folder and prints to the terminal if they don't have 3 channels and/or not RGB.

    Args:
        final_size: The size the images should be.
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
        if im_size != (final_size, final_size):
            print("Wrong size: ", im_size)

if __name__ == '__main__':
    final_size = 128
    clean_image_data(final_size, "data/images/")
    #check_images(final_size)





            








