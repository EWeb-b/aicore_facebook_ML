# aicore_facebook_ML

## Exploring the Dataset
Created two Python files to clean the Products.csv and Images. 

# Tabular Data
- Stripped the '£' sign and commas from the price column.
- Converted the price column into a float64 type.
- Converted the 'category' column type into the category column type. This is a numeric type under the surface and so is faster to use etc.
- Ran the `convert_dtypes` command on the remainder of the columns so that they were automatically converted into their optimum column types.

![plot](readme_images/clean_tabular_data_1.png)

# Image Data
- Wrote the `clean_image_data` function which takes in the filepath of the images to be cleaned and then passes them to another function called `resize_image`. This function resizes the images to 512x512 pixels and also casts them as the RGB type of image. This means that they all have 3 channels as standard.
- The images are then saved by the `clean_image_data` in the cleaned_images directory.
- Wrote a small helper function `check_images` which simply loops through the cleaned images and checks that they're all of the correct size and all have 3 channels.

![plot2](readme_images/clean_images_1.png)