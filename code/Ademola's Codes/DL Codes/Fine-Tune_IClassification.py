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
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plt.show()

# # Initialize the Pretrained Model
# feature_extractor = ResNet50(weights='imagenet',
#                              input_shape=(512, 512, 3),
#                              input_shape=(None, None, 3),
#                              include_top=False)
#
# # Set this parameter to make sure it's not being trained
# feature_extractor.trainable = False
#
# # Set the input layer
# input_ = tf.keras.Input(shape=(512, 512, 3))
# input_ = tf.keras.Input(shape=(None, None, 3))
#
# # Set the feature extractor layer
# x = feature_extractor(input_, training=False)
#
# # Set the pooling layer
#
#
# x = tfa.layers.SpatialPyramidPooling2D(bins=[1, 2, 4])(x)
# # x = tf.keras.layers.GlobalAveragePooling2D()(x)
# x = Flatten(name="flatten")(x)
#
# # Add anything after here
#
# x = Dense(2048, activation="relu")(x)
# x = Dropout(0.25)(x)
# x = Dense(1024, activation="relu")(x)
# x = Dropout(0.25)(x)
# x = Dense(256, activation="relu")(x)
# x = Dropout(0.25)(x)
#
#
# # Set the final layer with sigmoid activation function
# output_ = tf.keras.layers.Dense(1, activation="sigmoid")(x)
# # output_ = Dense(len(CLASSES), activation="softmax")(x)
# # Create the new model object
# model = tf.keras.Model(input_, output_)

# # Compile it
# opt = tf.keras.optimizers.Adam(lr=1e-4, decay=5e-4)
# # opt = tf.keras.optimizers.SGD(lr=1e-4, momentum=0.9, decay=5e-4)
# model.compile(optimizer=opt,
#              loss='binary_crossentropy',
#              metrics=['accuracy'])
#
# # Print The Summary of The Model
# model.summary()





# # def build_model():
# base = ResNet50(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
# base.trainable = False
# headModel = base.output
# headModel = tf.keras.layers.GlobalAveragePooling2D()(headModel)
# headModel = Dense(2048, activation="relu")(headModel)
# headModel = Dropout(0.25)(headModel)
# headModel = Dense(1024, activation="relu")(headModel)
# headModel = Dropout(0.25)(headModel)
# headModel = Dense(512, activation="relu")(headModel)
# headModel = Dropout(0.25)(headModel)
# headModel = Dense(1, activation="sigmoid")(headModel)
#
# model = Model(inputs=base.input, outputs=headModel)
# model.summary()
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


# model_use = Model(inputs=base.input, outputs=headModel)

# opt = tf.keras.optimizers.Adam(lr=1e-5)#, decay=0.0001/20)
# model_use.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# Prepare the train, valid and test directories for the generator
target_size = (512, 512)

# Create object of ImageDataGenerator
trainAug = ImageDataGenerator(featurewise_center=True,
                              featurewise_std_normalization=True,
                              rotation_range=20,
                              width_shift_range=0.2,
                              height_shift_range=0.2,
                              horizontal_flip=True)


valAug = ImageDataGenerator(featurewise_center=True,
                            featurewise_std_normalization=True,
                            rotation_range=20,
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            horizontal_flip=True)

mean = np.array([123.68, 116.779, 103.939], dtype="float32")
trainAug.mean = mean
valAug.mean = mean

trainGen = trainAug.flow_from_directory(
    train_directory,
    class_mode="binary",
    target_size=(512, 512),
    color_mode="rgb",
    shuffle="True",
    batch_size=32)

valGen = valAug.flow_from_directory(
    val_directory,
    class_mode="binary",
    target_size=(512, 512),
    color_mode="rgb",
    shuffle=False,
    batch_size=32)

testGen = valAug.flow_from_directory(
    test_directory,
    class_mode="binary",
    target_size=(512, 512),
    color_mode="rgb",
    shuffle=False,
    batch_size=32)


# my_callbacks = [
#     tf.keras.callbacks.ModelCheckpoint(monitor='val_accuracy', mode='max', save_best_only=True,
#                                        filepath='/media/ikusanaa/runImageClassification/model.{epoch:02d}-{val_loss:.2f}.h5'),
#     tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.001, min_delta=0.0001)
# ]

# # Training the model
# # model.fit_generator(trainGen, epochs=40, validation_data=valGen)
# model.fit(trainGen, epochs=40, validation_data=valGen, callbacks=my_callbacks)
# # Save model
# print("[INFO] saving model...")
# model.save('trained_classification', save_format="h5")
#
# # Load the saved model
# print("[INFO] loading saved model...")
# model = tf.keras.models.load_model('trained_classification_4')
#
# opt = tf.keras.optimizers.Adam(lr=5e-5)
# model.compile(optimizer=opt,
#              loss='binary_crossentropy',
#              metrics=['accuracy'])
#
#
# # Re-Training the model
# model.fit(trainGen, epochs=20, validation_data=valGen)
#
# # Saving the retrained model
# print("[INFO] saving model...")
# model.save('trained_classification_2', save_format="h5")

my_callbacks = [
    tf.keras.callbacks.ModelCheckpoint(monitor='val_accuracy', mode='max', save_best_only=True,
                                       filepath='/media/ikusanaa/runImageClassification/model.{epoch:02d}-{val_loss:.2f}.h5'),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.00001, min_delta=0.0001)
]

my_callbacks_2 = [
    tf.keras.callbacks.ModelCheckpoint(monitor='val_accuracy', mode='max', save_best_only=True,
                                       filepath='/media/ikusanaa/runImageClassification/model_2.{epoch:02d}-{val_loss:.2f}.h5'),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.00000000001, min_delta=0.0001)
]

# print("[INFO] loading saved model...")
# model = tf.keras.models.load_model('/media/ikusanaa/runImageClassification/model_8363.h5')
#
# opt = tf.keras.optimizers.Adam(lr=1e-5)
# model.compile(optimizer=opt,
#              loss='binary_crossentropy',
#              metrics=['accuracy'])
#
#
# # Re-Training the model
# model.fit(trainGen, epochs=20, validation_data=valGen, callbacks=my_callbacks)
#
# # Saving the retrained model
# print("[INFO] saving model...")
# model.save('trained_classification_6', save_format="h5")


print("[INFO] loading saved model...")
model = tf.keras.models.load_model('/media/ikusanaa/runImageClassification/model_8689.h5')


# for layer in feature_extractor.layers[15:]:
#     layer.trainable = True
#
#
# # loop over the layers in the model and show which ones are trainable
# # or not
# for layer in feature_extractor.layers:
#     print("{}: {}".format(layer, layer.trainable))


opt = tf.keras.optimizers.Adam(lr=1e-12)
model.compile(optimizer=opt,
             loss='binary_crossentropy',
             metrics=['accuracy'])

# Re-Training the model
H = model.fit(trainGen, epochs=10, validation_data=valGen, callbacks=my_callbacks_2)

# reset the testing generator and then use our trained model to
# make predictions on the data
print("[INFO] evaluating network...")
testGen.reset()
test_loss, test_acc = model.evaluate(testGen)
print(test_loss, test_acc)

plot_loss(H)
