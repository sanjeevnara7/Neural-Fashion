# Neural-Fashion
Neural Fashion Captioning &amp; Attribute Generation using Transformer Networks

## AttributeClassifier (jupyter notebook)
This notebook contains the code to initialize and train *resnet34* and *swin_t* models for predicting the attributes of an image in the **Fashion** dataset. The *ImageNet* classifier of the model is replaced by our *18 Classifier* layers each one for predicting the value of each attribute respectively. In the $1^{st}$ phase, we only train the classifier layers so that they adapt to the attributes in the dataset. In the $2^{nd}$ phase, we train some final modules of both models along with the classifier layers. In the *final* phase, we fully train both the networks. The training and validation loss and accuracies are recorded in the **tensorboard** for visualisation.

To compare **ResNet-152** with **Swin_Small**, simply change the models being loaded from **torchvision** : *resnet34* to *resnet152*, *swin_t* to *swin_s* in this notebook. We trained these models separately in **Google Colab**, due to longer training times, so the related code is not present here.

## CaptionModel (jupyter notebook)
Here, we first load the images and related captions from the **Fashion** dataset. The corresponding vocabulary, already created by functions in *utils*, is also loaded and displayed. Along with the *swin_t* backbone, we create the *DecoderRNN* which takes the **feature** map predicted by the *swin_t* backbone and the **word** at **current time step** to predict the **next word**. Here the *swin_t* backbone remains **frozen**.

We compare the performance of models loaded with our *trained* *swin_t* weights, and the *ImageNet* weights.

## labels (folder)
### shape (folder)
Contains the shape attributes of images in the **Fashion** dataset.

### texture (folder)
Contains the fabric, pattern attributes of images in the **Fashion** dataset.

### captions.json
Contains the captions of images in the **Fashion** dataset.

### train_data.npy
Contains the training images along with their attributes.

### validation_data.npy
Contains the validation images along with their attributes.

### train_val_split.py
Contains the code utilized for splitting the train and validation samples from the full dataset.

## runs (folder)
Contains all the tensorboard runs related to our trainings (helps in visualising a training). 

## tensorboard_screens (folder)
Contains the screenshots of the tensorboard runs corresponding to the *attribute* prediction trainings, and *caption* generation trainings.

## utils (folder)
### BLEU.py
This is helpful for computing the BLEU score between predicted and ground-truth caption.

### customDataset.py
This contains code for loading and preparing the **Fashion** dataset. We return the image along with its attributes, and caption.

### load_funcs.py
This contains code for reading the dataset and preparing the dataloader. It also has code for preprocessing the captions of images.

### train_funcs.py
This contains the *fit_classifier* function to train the *attribute* prediction model. It has *fit* function to train the *caption* generation model. It also contains other helper functions for computing losses and getting the predictions from those models.

## PureT_Fashion folder
This folder basically contains the modified [Pure Transformer](https://github.com/232525/PureT) model which takes feature inputs from the attribute trained **Swin_Tiny** transfomer model for improving the interactions between captions and image to generate semantically more meaningful caption. We couldn't train this model and compare with **LSTM based** decoder, and so we kept it as part of **future work**

## Github code link
Here is the link to access our code --> [Neural Fashion](https://github.com/sanjeevnara7/Neural-Fashion)
