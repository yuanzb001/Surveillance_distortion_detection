# import keras.losses
from tensorflow.keras.applications import ResNet50
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import Callback
# from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from keras.optimizers import adam_v2
import cv2
# from scipy.stats import pearsonr
# from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
# from keras import models, layers
import pandas as pd
import shutil
import re
import glob
import os
import matplotlib.pyplot as plt
# import theano

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


def m_std(mn, std, values):
    results = []
    diff2 = mn
    # diff3 = abs(mn + std)
    # print(diff3)


    for value in values:
        if value > 0:
            result = "Satisfactory"

        # elif diff2 <= value <= diff3:
        #     result = "Good"

        else:
            result = "Unsatisfactory"

        results.append(result)

    return results


iqa_score = 1
image_size = 224
CLASSES = ["Unsatisfactory", "Satisfactory"]


# Directory paths of the scores and images
# parent_path = "/media/ikusanaa/runObjectDetection/Object_Detection/"
# score_path = "/media/ikusanaa/runObjectDetection/mAP_scores_2/"

parent_path = "/media/ikusanaa/runImageClassification/Dataset/"
score_path = "/media/ikusanaa/runImageClassification/Score_cTE/"

# Directory for both images and scores
dataset_list_score = glob.glob(score_path + "/**/*.txt", recursive=True)
# images_path = glob.glob(parent_path + "/**/*.jpg", recursive=True)
images_path = glob.glob(parent_path + "/**/*.JPEG", recursive=True)

# Dataframe from the dataset image paths
df_dataset = pd.DataFrame(images_path)
# Combining all the IoU scores and create a dataframe
df_score = pd.DataFrame()
for each_score in dataset_list_score:
    df_each_score = pd.read_csv(each_score)
    df_score = pd.concat([df_score, df_each_score], ignore_index=True, axis=0)
# df_score.pop('filename')

# Name the column of the dataframe for IoU scores
df_score.columns = ['cTE']
mn = df_score['cTE'].mean()
std = df_score['cTE'].std()


# Changing the values to categorical
df_score['cTE'] = m_std(mn, std, df_score['cTE'])
# Adding the column to the dataset dataframe
df_dataset[len(df_dataset.columns)] = df_score['cTE']
df_dataset.columns = ['image_path', 'cTE']

# Defining the training parameters
train_ratio = 0.80
validation_ratio = 0.10
test_ratio = 0.10

# Train dataset is now 80% of the entire dataset
X_train, X_test, y_train, y_test = train_test_split(df_dataset['image_path'], df_dataset['cTE'],
                                                    test_size=1-train_ratio)

# Test and Validation now 10% each
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio))

# Changing the outputs to a numpy array
X_train = X_train.to_numpy()
X_val = X_val.to_numpy()
X_test = X_test.to_numpy()

y_train = y_train.to_numpy()
y_val = y_val.to_numpy()
y_test = y_test.to_numpy()

print(len(X_train))
print(len(y_train))
print(len(X_val))
print(len(y_val))
print(len(X_test))
print(len(y_test))


# # Create a training and testing directory
mode = 0o777
# train_directory = "/media/ikusanaa/runObjectDetection/train_dir"
# test_directory = "/media/ikusanaa/runObjectDetection/test_dir"
# val_directory = "/media/ikusanaa/runObjectDetection/val_dir"

train_directory = "/media/ikusanaa/runImageClassification/train_dir"
test_directory = "/media/ikusanaa/runImageClassification/test_dir"
val_directory = "/media/ikusanaa/runImageClassification/val_dir"

datasets = [("training", X_train, y_train, train_directory),
            ("validation", X_val, y_val, val_directory),
            ("testing", X_test, y_test, test_directory)
            ]

for (dType, imagePaths, classes, baseOutput) in datasets:
    # show which data split we are creating
    print("[INFO] building '{}' split".format(dType))
    # if the output base output directory does not exist, create it
    if not os.path.exists(baseOutput):
        print("[INFO] 'creating {}' directory".format(baseOutput))
        os.makedirs(baseOutput)
#     # loop over the input image paths
    for i in range(len(imagePaths)):
        # extract the filename of the input image along with its
        # corresponding class label
        filename = imagePaths[i].split(os.path.sep)
        filename = ''.join(filename[5:])
        # filename = imagePaths[i].split(os.path.sep)[-1]
        label = classes[i]
        # build the path to the label directory
        labelPath = os.path.sep.join([baseOutput, label])
#         # print(labelPath)
        # if the label output directory does not exist, create it
        if not os.path.exists(labelPath):
            print("[INFO] 'creating {}' directory".format(labelPath))
            os.makedirs(labelPath)
        # construct the path to the destination image and then copy
        # the image itself
        p = os.path.sep.join([labelPath, filename])
        print(p)

        shutil.copy2(imagePaths[i], p)

# # Create directories
# if os.path.isdir(train_directory):
#     # delete the directory
#     shutil.rmtree(train_directory)
#     # remake the directory
#     os.mkdir(train_directory, mode)
# else:
#     # make the directory
#     os.mkdir(train_directory, mode)
#
#
# if os.path.isdir(test_directory):
#     # delete the directory
#     shutil.rmtree(test_directory)
#     # remake the directory
#     os.mkdir(test_directory, mode)
# else:
#     # make the directory
#     os.mkdir(test_directory, mode)
#
#
# if os.path.isdir(val_directory):
#     # delete the directory
#     shutil.rmtree(val_directory)
#     # remake the directory
#     os.mkdir(val_directory, mode)
# else:
#     # make the directory
#     os.mkdir(val_directory, mode)
#
#
# # Training dataset creation
# number_file = 0
# for index, value in X_train.items():
#     number_file += 1
#     number_file = str(number_file)
#     number_file = number_file.zfill(5)
#     full_new_path = train_directory + "/" + number_file + '.jpg'
#     print(full_new_path)
#     number_file = int(number_file)
#     shutil.copy(value, full_new_path)
#
#
# # Validation dataset creation
# number_file = 0
# for index, value in X_val.items():
#     number_file += 1
#     number_file = str(number_file)
#     number_file = number_file.zfill(5)
#     full_new_path = val_directory + "/" + number_file + '.jpg'
#     print(full_new_path)
#     number_file = int(number_file)
#     shutil.copy(value, full_new_path)
#
#
# # Testing dataset creation
# number_file = 0
# for index, value in X_test.items():
#     number_file += 1
#     number_file = str(number_file)
#     number_file = number_file.zfill(5)
#     full_new_path = test_directory + "/" + number_file + '.jpg'
#     print(full_new_path)
#     number_file = int(number_file)
#     shutil.copy(value, full_new_path)


# # Loading training data
# def load_train(path_dir, y_data):
#     image_pro = 0
#     print('[INFO] Working on this {}', path_dir)
#     dataset_list = sorted(os.listdir(path_dir), key=natural_keys)
#     X_data = []
#
#     for image in dataset_list:
#         img = cv2.imread(os.path.join(path_dir, image))
#         img = cv2.resize(img, (image_size, image_size))
#         image_pro += 1
#         print("[INFO] Working on Image: {}".format(image_pro))
#         X_data.append(img)
#
#     return X_data, y_data


# # Read and Normalize Data
# def read_and_normalize_train_data(path_dir, y_data):
#     train_data, train_target = load_train(path_dir, y_data)
#     train_data = np.array(train_data, dtype=np.float32)
#     train_target = np.array(train_target, dtype=np.str)
#     m = train_data.mean()
#     s = train_data.std()
#
#     print('Train- mean:{}, sd:{}'.format(m, s))
#     train_data -= m
#     train_data /= s
#     print('Train Shape: {}'.format(train_data.shape))
#     print(train_data.shape[0], ' train samples')
#     return train_data, train_target


class PrintDot(Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            print('')
        print('.', end='')


def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 5])
    plt.xlabel('Epoch')
    plt.ylabel('Error [IoU]')
    plt.legend()
    plt.grid(True)
    plt.show()


# def build_model():
base = ResNet50(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

headModel = base.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(256, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(len(CLASSES), activation="softmax")(headModel)

model = Model(inputs=base.input, outputs=headModel)
model.summary()
# for layer in base.layers:
#     layer.trainable = False
#
#
#
# model = models.Sequential()
# model.add(base)
# model.add(layers.Flatten())
# model.add(Dense(2048, activation='relu'))
# model.add(Dense(1024, activation='relu'))
# model.add(Dense(iqa_score))
# opt = tf.keras.optimizers.Adam(lr=0.001)
# model.compile(loss="mean_squared_error", optimizer=opt, metrics=[keras.losses.MeanSquaredError(),
#                                                                  keras.losses.MeanAbsoluteError()])
# return model


model_use = Model(inputs=base.input, outputs=headModel)

for layer in base.layers:
    layer.trainable = False

opt = tf.keras.optimizers.Adam(lr=0.0001, decay=0.0001/20)
model_use.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])


# Prepare the train, valid and test directories for the generator
target_size = (224, 224)

# Create object of ImageDataGenerator
trainAug = ImageDataGenerator(rotation_range=25,
                              shear_range=0.2,
                              horizontal_flip=True,
                              fill_mode="nearest")

valAug = ImageDataGenerator()

mean = np.array([123.68, 116.779, 103.939], dtype="float32")
trainAug.mean = mean
valAug.mean = mean

trainGen = trainAug.flow_from_directory(
    train_directory,
    class_mode="categorical",
    target_size=(224, 224),
    color_mode="rgb",
    shuffle="True",
    batch_size=32)

valGen = valAug.flow_from_directory(
    val_directory,
    class_mode="categorical",
    target_size=(224, 224),
    color_mode="rgb",
    shuffle=False,
    batch_size=32
)

testGen = valAug.flow_from_directory(
    test_directory,
    class_mode="categorical",
    target_size=(224, 224),
    color_mode="rgb",
    shuffle=False,
    batch_size=32
)

# X_trn, y_trn = read_and_normalize_train_data(train_directory, y_train)
# X_tst, y_tst = read_and_normalize_train_data(test_directory, y_test)
# X_vl, y_vl = read_and_normalize_train_data(val_directory, y_val)

# Training the model
H = model_use.fit_generator(trainGen, epochs=50, validation_data=valGen, validation_steps=len(X_val) // 32,
                            steps_per_epoch=len(X_train) // 32)


# # reset the testing generator and then use our trained model to
# # make predictions on the data
# print("[INFO] evaluating network...")
# testGen.reset()
# predIdxs = model_use.predict_generator(testGen, steps=(len(X_test) // 32) + 1)
# # for each image in the testing set we need to find the index of the
# # label with corresponding largest predicted probability
#
# predIdxs = np.argmax(predIdxs, axis=1)
# # show a nicely formatted classification report
# print(classification_report(testGen.classes, predIdxs, target_names=testGen.class_indices.keys()))


# reset our data generators
trainGen.reset()
valGen.reset()


# now that the head FC layers have been trained/initialized, lets unfreeze the final set of CONV layers and make them
# trainable

for layer in base.layers[15:]:
    layer.trainable = True

for layer in base.layers:
    print("{}: {}".format(layer, layer.trainable))


opt = tf.keras.optimizers.Adam(lr=0.0001, decay=0.0001/20)
model_use.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])


# Retraining the model
H = model_use.fit_generator(trainGen, epochs=20, validation_data=valGen, validation_steps=len(X_val) // 32,
                            steps_per_epoch=len(X_train) // 32)



# Save model
print("[INFO] saving model...")
model_use.save('trained_mclass_cTE', save_format="h5")

# reset the testing generator and then use our trained model to
# make predictions on the data
print("[INFO] evaluating network...")
testGen.reset()
predIdxs = model_use.predict_generator(testGen, steps=(len(X_test) // 32) + 1)

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)
# show a nicely formatted classification report
print(classification_report(testGen.classes, predIdxs, target_names=testGen.class_indices.keys()))


