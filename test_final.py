from __future__ import print_function

# Networks

from keras.applications.vgg16 import VGG16
import utils

# Utils
import numpy as np
import argparse

import os
import cv2
import time


# For boolean input from the command line
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# Command line args
parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs to train for')
parser.add_argument('--mode', type=str, default="predict", help='Select "train", or "predict" mode. \
    Note that for prediction mode you have to specify an image to run the model on.')
parser.add_argument('--continue_training', type=str2bool, default=False,
                    help='Whether to continue training from a checkpoint')
parser.add_argument('--dataset', type=str, default="Dataset", help='Dataset you are using.')
parser.add_argument('--resize_height', type=int, default=224, help='Height of cropped input image to network')
parser.add_argument('--resize_width', type=int, default=224, help='Width of cropped input image to network')
parser.add_argument('--batch_size', type=int, default=32, help='Number of images in each batch')
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
BATCH_SIZE = args.batch_size
WIDTH = args.resize_width
HEIGHT = args.resize_height
FC_LAYERS = [1024, 1024]
preprocessing_function = None
base_model = None


if args.model == "VGG16":
    from keras.applications.mobilenet import preprocess_input

    preprocessing_function = preprocess_input
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))

else:
    ValueError("The model you requested is not supported in Keras")

haar_file = 'haarcascade_frontalface_default.xml'

face_cascade = cv2.CascadeClassifier(haar_file)
###################################################33##difference###################################################
if args.mode == "predict":

  
    image = cv2.imread("Dataset 1/iv/iv2.JPG")
    
    im=image

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)

    if len(faces)>0:

      # Store pic of the face
      for (x, y, w, h) in faces:
          cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
          face = gray[y:y + h, x:x + w]

          cv2.imwrite("test.jpg",face)


          image=cv2.imread("test.jpg")


          image = np.float32(cv2.resize(image, (224, 224)))

          print(np.shape(image))
          image = preprocessing_function(image.reshape(1, HEIGHT, WIDTH, 3))

          class_list_file = "./checkpoints1/" + args.model + "_" + args.dataset + "_class_list.txt"

          class_list = utils.load_class_list(class_list_file)

          # print(base_model)
          # print(class_list)
          #
          FC_LAYERS = [1024, 1024]
          dropout = 0.5
          finetune_model = utils.build_finetune_model(base_model,
                                                      dropout=dropout,
                                                      fc_layers=FC_LAYERS,
                                                      num_classes=len(class_list))
          print(finetune_model)
          finetune_model.load_weights("./checkpoints1/" + args.model + "_model_weights.h5")

          # Run the classifier and print results
          st = time.time()

          out = finetune_model.predict(image)

          confidence = out[0]
          class_prediction = list(out[0]).index(max(out[0]))
          class_name = class_list[class_prediction]

          run_time = time.time() - st

          print("Predicted class = ", class_name)
          print("Confidence = ", confidence)
          print("Run time = ", run_time)
          
          if confidence[class_prediction] > 0.53:  
              cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)
              cv2.putText(im,class_name[0] , (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN,
                        1, (0, 255, 0))

              cv2.putText(im, str(confidence[class_prediction]), (x - 20, y - 20), cv2.FONT_HERSHEY_PLAIN,1, (0, 255, 0))

          else:
                cv2.putText(im, "unknown", (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN,
                            1, (0, 255, 0))

     

    else:
         print("face not found")

    cv2.imwrite("output.png",im)

    import matplotlib.pyplot as plt
    plt.imshow(im)
    plt.show()


