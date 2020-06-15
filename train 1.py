from __future__ import print_function   #to bring the print function from Python 3 into Python 2.6+

# Networks
from keras.applications.vgg16 import VGG16   #feature extraction
from keras.preprocessing.image import ImageDataGenerator   #The Keras deep learning neural network library
                 # provides the capability to fit models using image data augmentation via the ImageDataGenerator class.
                 #  Accepting a batch of images or importing images with label used for training

# Layers
from keras import backend as K    #K. function takes the input and output tensors as list so that you can create
                                   # a function from many input to many output.

# Other
from keras.optimizers import SGD, Adam #stochastic gradient descent,Adaptive Moment Estimation
                                       # is the "classical" optimization algorithms
from keras.callbacks import ModelCheckpoint, LearningRateScheduler   #callback is a powerful tool to customize
                                                       # the behavior of a Keras model during training, evaluation,
                                                       # or inference, including reading/changing the Keras model.
                                                    #model is automatically saved during training,
                                                    #sets the learning rate according to schedule.(schedule: a function
                                                    # that takes an epoch index(integer, indexed from 0) and current learning rate
                                                    #as inputs and returns a new learning rate as output (float)
from sklearn.metrics import classification_report, confusion_matrix   #A Classification report is used to measure the  quality of
                                                # predictions from a classification algorithm. More specifically, True Positives,
                                              # False Positives, True negatives and False Negatives are used to predict the metrics
                                              # of a classification report . The classification_report function builds a text
                                            # report showing the main classification metrics while A confusion matrix is a summary
                                            # of prediction results on a classification problem
# Utils
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os, sys, csv

# Files
import utils


# For boolean input from the command line
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# Command line args
# 1 Epoch = 1 Forward pass + 1 Backward pass for ALL training samples. no of learning times
# If we have 1000 training samples and Batch size is set to 500, it will take 2 iterations to complete 1 Epoch.
#	epoch -- number of epochs ( default = 55 iterations) which decides the number of times the learning should be done.
#	dataset – set the default location of the stored image.
#	resize_height and resize_width – to set default to 224x224.
#	batch_size – the total number of images per batch for training.
#	zoom,shear,etc. are set to false

parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, default=55, help='Number of epochs to train for')
parser.add_argument('--mode', type=str, default="train", help='Select "train", or "predict" mode. \
    Note that for prediction mode you have to specify an image to run the model on.')
parser.add_argument('--continue_training', type=str2bool, default=False,
                    help='Whether to continue training from a checkpoint')
parser.add_argument('--dataset', type=str, default="Dataset2", help='Dataset you are using.')
parser.add_argument('--resize_height', type=int, default=224, help='Height of cropped input image to network')
parser.add_argument('--resize_width', type=int, default=224, help='Width of cropped input image to network')
parser.add_argument('--batch_size', type=int, default=127, help='Number of images in each batch')
parser.add_argument('--dropout', type=float, default=1e-3, help='Dropout ratio')
parser.add_argument('--h_flip', type=str2bool, default=False,
                    help='Whether to randomly flip the image horizontally for data augmentation')
parser.add_argument('--v_flip', type=str2bool, default=False,
                    help='Whether to randomly flip the image vertically for data augmentation')
parser.add_argument('--rotation', type=float, default=0.0,
                    help='Whether to randomly rotate the image for data augmentation')
parser.add_argument('--zoom', type=float, default=0.0, help='Whether to randomly zoom in for data augmentation')
parser.add_argument('--shear', type=float, default=0.0, help='Whether to randomly shear in for data augmentation')
parser.add_argument('--model', type=str, default="VGG16", help='Your pre-trained classification model of choice')
args = parser.parse_args()

# Global settings

BATCH_SIZE = args.batch_size   #Number of training samples in 1 Forward/1 Backward pass
WIDTH = args.resize_width
HEIGHT = args.resize_height
FC_LAYERS = [1024, 1024]
TRAIN_DIR = args.dataset + "/train/"  #to train the faces detected by the HaarClassif
VAL_DIR = args.dataset + "/eval/"   #to evaluate the trained images in every epoch so as to increase the accuracy

preprocessing_function = None
base_model = None

# Prepare the model

if args.model == "VGG16":
    from keras.applications.mobilenet import preprocess_input

    preprocessing_function = preprocess_input
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))  #load the vgg16 model
                                                                                           # (images height and width)

else:
    ValueError("The model you requested is not supported in Keras")

if args.mode == "train":   #starts training
    print("\n***** Begin training *****")
    print("Dataset -->", args.dataset)   # dataset – set the default location of the stored image.
    print("Model -->", args.model)       # args.model == "VGG16"
    print("Resize Height -->", args.resize_height)  # resize_height and resize_width – to set default to 224x224.
    print("Resize Width -->", args.resize_width)
    print("Num Epochs -->", args.num_epochs)   #55
    print("Batch Size -->", args.batch_size)   # 8- the total number of images per batch for training.

   # All are set to false
    print("Data Augmentation:")
    print("\tVertical Flip -->", args.v_flip)
    print("\tHorizontal Flip -->", args.h_flip)
    print("\tRotation -->", args.rotation)
    print("\tZooming -->", args.zoom)
    print("\tShear -->", args.shear)
    print("")

    # Create directories if needed
    if not os.path.isdir("checkpoints1"):
        os.makedirs("checkpoints1")

    # Prepare data generators.
    # Generate batches of tensor image data with real-time data augmentation.
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocessing_function, # The function will run after the image is resized and augmented.
        rotation_range=args.rotation, # Int. Degree range for random rotations.
        shear_range=args.shear, #Float. Shear Intensity
        zoom_range=args.zoom,   #Float. Range for random zoom.
        horizontal_flip=args.h_flip, #Boolean. Randomly flip inputs horizontally.
        vertical_flip=args.v_flip #Boolean. Randomly flip inputs vertically.
    )

    val_datagen = ImageDataGenerator(preprocessing_function=preprocessing_function)
      # flow_from- identify classes automatically from the folder name
    train_generator = train_datagen.flow_from_directory(TRAIN_DIR, target_size=(HEIGHT, WIDTH), batch_size=BATCH_SIZE)
   # TRAIN_DIR - train detected face, VAL_DIR - evaluate trained images .. sujith aanoo, graph
    validation_generator = val_datagen.flow_from_directory(VAL_DIR, target_size=(HEIGHT, WIDTH), batch_size=BATCH_SIZE)
    # eg : Found 212 images belonging to 2 classes.



    # Save the list of classes for prediction mode later
    class_list = utils.get_subfolders(TRAIN_DIR)

    print(len(class_list))

    print(",,,,,,,,,,,,,,,,,,")
    utils.save_class_list(class_list, model_name=args.model, dataset_name=args.dataset)
   # Fine tuning is a process to take a network model that has already been trained for a given task, and make it perform
    # a second similar task.
    finetune_model = utils.build_finetune_model(base_model, dropout=args.dropout, fc_layers=FC_LAYERS,
                                                num_classes=len(class_list))

    if args.continue_training:
        finetune_model.load_weights("./checkpoints1/" + args.model + "_model_weights.h5")

    adam = Adam(lr=0.00001) #  If our training is bouncing a lot on epochs then we need to decrease the learning rate
                            # so that we can reach global minima. slow learning greater accuracy
    finetune_model.compile(adam, loss='categorical_crossentropy', metrics=['accuracy'])
    # The purpose of loss functions is to compute the quantity that a model should seek to minimize during training.
    # Computes the crossentropy loss between the labels and predictions.
    # one-hot representation

# to get number of trained images and number of valid images
    num_train_images = utils.get_num_files(TRAIN_DIR)
    num_val_images = utils.get_num_files(VAL_DIR)


    def lr_decay(epoch):
        if epoch % 20 == 0 and epoch != 0:
            lr = K.get_value(model.optimizer.lr)
            K.set_value(model.optimizer.lr, lr / 2)
            print("LR changed to {}".format(lr / 2))
        return K.get_value(model.optimizer.lr)



    learning_rate_schedule = LearningRateScheduler(lr_decay)

    filepath = "./checkpoints1/" + args.model + "_model_weights.h5"
    checkpoint = ModelCheckpoint(filepath, monitor=["acc"], verbose=1, mode='max')
    callbacks_list = [checkpoint]

    H= finetune_model.fit_generator(train_generator, epochs=args.num_epochs, workers=8,
                                           steps_per_epoch=num_train_images // BATCH_SIZE,
                                           validation_data=validation_generator,
                                           validation_steps=num_val_images // BATCH_SIZE, class_weight='auto',
                                           shuffle=True, callbacks=callbacks_list)
    
    
    #Confusion Matrix and Classification Report
    Y_pred = finetune_model.predict_generator(validation_generator, num_val_images // BATCH_SIZE+1)
    y_pred = np.argmax(Y_pred, axis=1)
    print('Confusion Matrix')
    print(confusion_matrix(validation_generator.classes, y_pred))
    print('Classification Report')
    target_names = ['Abhishek', 'Aishwarya', 'Alisha', 'Ambadi','Anujith', 'Anusha', 'Arjun R', 'Arunima', 'Gayathri K', 'Potty', 'Reshma', 'Vaishnavi', 'Vishal', 'Unknown Class']
    print(classification_report(validation_generator.classes, y_pred, target_names=target_names))
    
    
    
    
    
    
    
    # args["plot"] = "plot1.png"
    # plot the training loss and accuracy
    N = args.num_epochs
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig("plot1.png")
