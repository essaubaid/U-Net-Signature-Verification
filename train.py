import os
import numpy as np
from tqdm import tqdm
from skimage.io import imread, imshow
from skimage.transform import resize
from skimage.color import gray2rgb
from model import u_Net
import matplotlib.pyplot as plt
import tensorflow as tf

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNEL = 3

TRAIN_PATH = 'new_data/train/'
TEST_PATH = 'new_data/test/'

train_ids = os.listdir(TRAIN_PATH + 'image/')
test_ids = os.listdir(TEST_PATH + 'image/')

x_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH,
                   IMG_CHANNEL), dtype=np.uint8)
y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

# print('X_train shape:', x_train.shape)
# print('Y_train shape:', y_train.shape)

for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = TRAIN_PATH
    img = imread(path + '/image/' + id_)
    if len(img.shape) == 2:  # Check if the image is grayscale
        img = gray2rgb(img)  # Convert grayscale image to RGB
    img = img[:, :, :IMG_CHANNEL]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH),
                 mode='constant', preserve_range=True)
    x_train[n] = img  # Fill empty X_train with values from img
    mask = imread(path + '/mask/' + id_)
    if len(mask.shape) == 2:  # Check if the mask is grayscale
        mask = gray2rgb(mask)  # Convert grayscale mask to RGB
    mask = mask[:, :, :1]
    mask = resize(mask, (IMG_HEIGHT, IMG_WIDTH),
                  mode='constant', preserve_range=True)
    y_train[n] = mask

# test images
x_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH,
                  IMG_CHANNEL), dtype=np.uint8)
sizes_test = []
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = TEST_PATH
    img = imread(path + '/image/' + id_)[:, :, :IMG_CHANNEL]
    sizes_test.append([img.shape[0], img.shape[1]])
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH),
                 mode='constant', preserve_range=True)
    x_test[n] = img


model = u_Net()
# Modelcheckpoint
checkpointer = tf.keras.callbacks.ModelCheckpoint(
    'model_for_nuclei.h5', verbose=1, save_best_only=True)

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir='logs')]

results = model.fit(x_train, y_train, validation_split=0.1,
                    batch_size=16, epochs=25, callbacks=callbacks)
