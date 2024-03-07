import keras.losses
import keras.backend
from tensorflow.keras.applications.resnet import ResNet50
import tensorflow_addons as tfa
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from sklearn.metrics import accuracy_score
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
import pandas as pd
import shutil
import re
import glob
import os
import matplotlib.pyplot as plt
# import theano
from sklearn.metrics import classification_report, confusion_matrix
import itertools

# theano.config.device = 'gpu'
# theano.config.floatX='float32'

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text

    return retval


def natural_keys(text):
    return [atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text)]


def spp_layer(input_, levels=(6, 3, 2, 1), name='SPP_layer'):
    shape = input_.shape
    print(shape)
    # with tf.keras.variable_scope(name):
    pyramid = []
    for n in levels:

        stride_1 = np.floor(float(shape[1] // n)).astype(np.int32)
        stride_2 = np.floor(float(shape[2] // n)).astype(np.int32)
        ksize_1 = stride_1 + (shape[1] % n)
        ksize_2 = stride_2 + (shape[2] % n)
        pool = tf.nn.max_pool(input_,
                              ksize=[1, ksize_1, ksize_2, 1],
                              strides=[1, stride_1, stride_2, 1],
                              padding='VALID')

        # print("Pool Level {}: shape {}".format(n, pool.get_shape().as_list()))
        pyramid.append(tf.reshape(pool, [shape[0], -1]))
        spp_pool = tf.concat(1, pyramid)
    return spp_pool


BATCH_SIZE = 32
image_size = 224
CLASSES = ["Excellent", "Bad"]

parent_path = "/media/ikusanaa/Distortion_Rate/dataset/"
score_path = "/media/ikusanaa/Distortion_Rate/score_path/result.txt"

# Directory for both images
images_path = sorted(glob.glob(parent_path + "/**/*.*", recursive=True), key=natural_keys)

# Dataframe from the dataset image paths
df_dataset = pd.DataFrame(images_path)

# Name the column of the dataframe
df_score = pd.read_csv(score_path, header=0)

# Adding the column to the dataset dataframe
df_dataset[len(df_dataset.columns)] = df_score['Class']
df_dataset.columns = ['image_path', 'Class']

print(df_dataset)
df_dataset.to_csv("complete_result.csv", header=True, index=False)

# # Defining the training parameters
# train_ratio = 0.80
# validation_ratio = 0.10
# test_ratio = 0.10
#
# # Train dataset is now 80% of the entire dataset
# X_train, X_test, y_train, y_test = train_test_split(df_dataset['image_path'], df_dataset['Class'],
#                                                     test_size=1-train_ratio, random_state=3)
#
# # Test and Validation now 10% each
# X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio), random_state=3)
#
# # Changing the outputs to a numpy array
# X_train = X_train.to_numpy()
# X_val = X_val.to_numpy()
# X_test = X_test.to_numpy()
#
# y_train = y_train.to_numpy()
# y_val = y_val.to_numpy()
# y_test = y_test.to_numpy()
#
# print(len(X_train))
# print(len(y_train))
# print(len(X_val))
# print(len(y_val))
# print(len(X_test))
# print(len(y_test))
#
# # Create a training and testing directory
# mode = 0o777
# #
# train_directory = "/media/ikusanaa/Distortion_Rate/train_dir"
# test_directory = "/media/ikusanaa/Distortion_Rate/test_dir"
# val_directory = "/media/ikusanaa/Distortion_Rate/val_dir"
#
# datasets = [("training", X_train, y_train, train_directory),
#             ("validation", X_val, y_val, val_directory),
#             ("testing", X_test, y_test, test_directory)
#             ]
#
#
# for (dType, imagePaths, classes, baseOutput) in datasets:
#     # show which data split we are creating
#     print("[INFO] building '{}' split".format(dType))
#     # if the output base output directory does not exist, create it
#     if not os.path.exists(baseOutput):
#         print("[INFO] 'creating {}' directory".format(baseOutput))
#         os.makedirs(baseOutput)
# #     # loop over the input image paths
#     for i in range(len(imagePaths)):
#         # extract the filename of the input image along with its
#         # corresponding class label
#         filename = imagePaths[i].split(os.path.sep)
#         filename = ''.join(filename[5:])
#         # filename = imagePaths[i].split(os.path.sep)[-1]
#         label = classes[i]
#         # build the path to the label directory
#         labelPath = os.path.sep.join([baseOutput, label])
# #         # print(labelPath)
#         # if the label output directory does not exist, create it
#         if not os.path.exists(labelPath):
#             print("[INFO] 'creating {}' directory".format(labelPath))
#             os.makedirs(labelPath)
#         # construct the path to the destination image and then copy
#         # the image itself
#         p = os.path.sep.join([labelPath, filename])
#         print(p)
#
#         shutil.copy2(imagePaths[i], p)
#
# class PrintDot(Callback):
#     def on_epoch_end(self, epoch, logs):
#         if epoch % 100 == 0:
#             print('')
#         print('.', end='')
#
#
# def plot_loss(history):
#     plt.plot(history.history['loss'], label='loss')
#     plt.plot(history.history['val_loss'], label='val_loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Error')
#     plt.legend()
#     plt.grid(True)
#     plt.show()
#
#
# # Plot the confusion matrix. Set Normalize = True/False
# def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
#     plt.figure(figsize=(20, 20))
#
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)
#
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         cm = np.around(cm, decimals=2)
#         cm[np.isnan(cm)] = 0.0
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, cm[i, j],
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")
#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     plt.show()
#
#
#
#
# # Initialize the Pretrained Model
# feature_extractor = ResNet50(weights='imagenet',
#                              input_shape=(224, 224, 3),
#                              # input_shape=(None, None, 3),
#                              include_top=False)
#
# # Set this parameter to make sure it's not being trained
#
#
# for layers in feature_extractor.layers:
#     layers.trainable = True
#
#
# resnet_x = Flatten()(feature_extractor.output)
# # resnet_x = Dense(256, activation='relu')(resnet_x)
# resnet_x = Dense(2048, activation="relu")(resnet_x)
# resnet_x = Dropout(0.25)(resnet_x)
# resnet_x = Dense(1024, activation="relu")(resnet_x)
# resnet_x = Dropout(0.25)(resnet_x)
# resnet_x = Dense(256, activation="relu")(resnet_x)
# resnet_x = Dropout(0.25)(resnet_x)
# resnet_x = Dense(2, activation='softmax')(resnet_x)
# resnet_x_final = Model(inputs=feature_extractor.input, outputs=resnet_x, name="FeatureExtractionModel")
# resnet_x_final.summary()
# opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.7)
# resnet_x_final.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])
#
#
# # Create object of ImageDataGenerator
# trainAug = ImageDataGenerator(rescale=1.0/255,
#                               shear_range=0.2,
#                               zoom_range=0.2)
#
#
# valAug = ImageDataGenerator(rescale=1.0/255)
# #
# # mean = np.array([123.68, 116.779, 103.939], dtype="float32")
# # trainAug.mean = mean
# # valAug.mean = mean
# #
# # trainGen = trainAug.flow_from_directory(
# #     train_directory,
# #     class_mode="categorical",
# #     target_size=(224, 224),
# #     shuffle="True",
# #     batch_size=32)
# #
# # valGen = valAug.flow_from_directory(
# #     val_directory,
# #     class_mode="categorical",
# #     target_size=(224, 224),
# #     shuffle=True,
# #     batch_size=32)
# #
# testGen = valAug.flow_from_directory(
#     test_directory,
#     class_mode="categorical",
#     target_size=(224, 224),
#     color_mode="rgb",
#     shuffle=False,
#     batch_size=32)
# #
# #
# # # Training the model
# # number_of_epochs = 60
# # resnet_filepath = 'resnetqa50'+'-saved-model-{epoch:02d}-val_acc-{val_acc:.2f}.hdf5'
# # resnet_checkpoint = tf.keras.callbacks.ModelCheckpoint(resnet_filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
# # resnet_early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
# # reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.05, patience=5, min_lr=0.000002)
# # callbacklist = [resnet_checkpoint, resnet_early_stopping, reduce_lr]
# # resnet_history = resnet_x_final.fit(trainGen, epochs=number_of_epochs, validation_data=valGen, callbacks=callbacklist, verbose=1)
#
# # print("[INFO] loading saved model...")
# # model = tf.keras.models.load_model('/media/ikusanaa/darknet/runObjectDetection/resnetqa50-saved-model-21-val_acc-0.92.hdf5')
# #
# # # reset the testing generator and then use our trained model to
# # # make predictions on the data
# # print("[INFO] evaluating network...")
# # testGen.reset()
# # # test_loss, test_acc = model.evaluate(testGen)
# # # print(test_loss, test_acc)
# #
# #
# #
# # test_score = model.evaluate_generator(testGen)
# # print("[INFO] accuracy: {:.2f}%".format(test_score[1] * 100))
# # print("[INFO] Loss: ", test_score[0])
# #
# # target_names = []
# # for key in testGen.class_indices:
# #     target_names.append(key)
# #
# # # print(target_names)
# #
# # #Confution Matrix
# #
# # Y_pred = model.predict_generator(testGen)
# # y_pred = np.argmax(Y_pred, axis=1)
# # print('Confusion Matrix')
# # cm = confusion_matrix(testGen.classes, y_pred)
# # plot_confusion_matrix(cm, target_names, title='Confusion Matrix')
# #
# # #Print Classification Report
# # print('Classification Report')
# # print(classification_report(testGen.classes, y_pred, target_names=target_names))
# #
