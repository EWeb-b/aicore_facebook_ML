# aicore_facebook_ML

# Exploring the Dataset
Created two Python files to clean the Products.csv and Images. 

## Tabular Data
- Stripped the '£' sign and commas from the price column.
- Converted the price column into a float64 type.
- Converted the 'category' column type into the category column type. This is a numeric type under the surface and so is faster to use etc.
- Ran the `convert_dtypes` command on the remainder of the columns so that they were automatically converted into their optimum column types.

![plot](readme_images/clean_tabular_data_1.png)

## Image Data
- Wrote the `clean_image_data` function which takes in the filepath of the images to be cleaned and then passes them to another function called `resize_image`. This function resizes the images to 512x512 pixels and also casts them as the RGB type of image. This means that they all have 3 channels as standard.
- The images are then saved by the `clean_image_data` in the cleaned_images directory.
- Wrote a small helper function `check_images` which simply loops through the cleaned images and checks that they're all of the correct size and all have 3 channels.

![plot2](readme_images/clean_images_1.png)

# Creating the Image Model
Created a Convolutional Neural Network (CNN) to categorise products based off their images. Then, used the pretrained ResNet-50 model instead to achieve better results by employing transfer learning.

## Creating the tranfer CNN
- The ResNet-50 model from Nvidia was used for the transfer learning.
- This classification problem has 13 different categories. By changing the final fully-connected layer of the ResNet-50 model, I was able to change it to start classifying products into one of these 13 different categories.

![plot](readme_images/transfer_CNN.png)

## Building the Training Loop
- The training function loops through each epoch, within which it loops over batches of data contained in the dataloaders, one for the training set, and one for the validation set.
- The model is trained and makes predictions during every batch of data from the dataloaders.
- The loss is also calculated during this time.

![plot](readme_images/images_train_loop.png)

## Saving the model
- By using a validation set of data, I was able to calculate the accuracy of my models as they trained.
- Then, by keeping track of the best model iteration, I could save that best performing model after the program had looped through every epoch.

![plot](readme_images/saving_best_image_model.png)

## Results
The results were visualised using tensorboard. Between each run I tweaked the model's hyperparameters, including the learning rate and batch size. Data augmentation was also applied to the input images to try and reduce the risk of overfitting, such as randomly flipping images and standardizing the images using their mean and standard deviation.
It's worth noting that the model overfitting was not overlooked at this stage. Knowing that the image model would be merged with the text model later, I chose to tackle any overfitting problems at that time instead.
- I used tensorboard to visualise the performance of the image model.
- The model could achieve a maximum of ~45% accuracy on the validation data, albeit with a large degree of overfitting on the data.
- This overfitting can be seen in the training loss converging towards 0 and the training accuracy reaching upwards of 90%, whilst the validation loss instead appeared to do nothing but increase over the training time.

![plot](readme_images/images_epoch_acc_train.png)
![plot](readme_images/images_epoch_loss_train.png)
![plot](readme_images/images_epoch_acc_val.png)
![plot](readme_images/images_epoch_loss_val.png)