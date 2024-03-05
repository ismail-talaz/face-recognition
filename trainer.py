import tensorflow as tf
import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import albumentations as alb
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, GlobalMaxPooling2D, Dropout
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import load_model
from mdl import FaceTracker

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)

def load_image(x): 
    byte_img = tf.io.read_file(x)
    img = tf.io.decode_jpeg(byte_img)
    return img

def load_labels(label_path):
    with open(label_path.numpy(), 'r', encoding = "utf-8") as f:
        label = json.load(f)
        
    return [label['class']], label['bbox']

def control(batch_size = 16,
            number_of_training_data=15000,
            optimizer = 'adam',
            sgd_momentum = 0.9,
            save_model = True,
            learning_rate = 0.00001,
            epoch = 10 ):
    train_images = tf.data.Dataset.list_files('aug_data\\train\\images\\*.jpg', shuffle=False)
    train_images = train_images.map(load_image)
    train_images = train_images.map(lambda x: tf.image.resize(x, (120,120)))
    train_images = train_images.map(lambda x: x/255)

    test_images = tf.data.Dataset.list_files('aug_data\\test\\images\\*.jpg', shuffle=False)
    test_images = test_images.map(load_image)
    test_images = test_images.map(lambda x: tf.image.resize(x, (120,120)))
    test_images = test_images.map(lambda x: x/255)

    val_images = tf.data.Dataset.list_files('aug_data\\val\\images\\*.jpg', shuffle=False)
    val_images = val_images.map(load_image)
    val_images = val_images.map(lambda x: tf.image.resize(x, (120,120)))
    val_images = val_images.map(lambda x: x/255)

    train_labels = tf.data.Dataset.list_files('aug_data\\train\\labels\\*.json', shuffle=False)
    train_labels = train_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))

    test_labels = tf.data.Dataset.list_files('aug_data\\test\\labels\\*.json', shuffle=False)
    test_labels = test_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))

    val_labels = tf.data.Dataset.list_files('aug_data\\val\\labels\\*.json', shuffle=False)
    val_labels = val_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))

    train = tf.data.Dataset.zip((train_images, train_labels))
    train = train.shuffle(number_of_training_data)
    train = train.batch(batch_size)
    train = train.prefetch(4)

    test = tf.data.Dataset.zip((test_images, test_labels))
    test = test.shuffle(number_of_training_data)
    test = test.batch(batch_size)
    test = test.prefetch(4)

    val = tf.data.Dataset.zip((val_images, val_labels))
    val = val.shuffle(number_of_training_data)
    val = val.batch(batch_size)
    val = val.prefetch(4)

    batches_per_epoch = len(train)
    lr_decay = (1./0.75 -1)/batches_per_epoch

    adm = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate, decay=lr_decay)
    sgd = tf.keras.optimizers.legacy.SGD(learning_rate=learning_rate, momentum=sgd_momentum, decay=lr_decay, nesterov=False)
    
    opt = adm if optimizer == 'adam' else sgd

    regressloss = localization_loss
    classloss = tf.keras.losses.BinaryCrossentropy()
    facetracker = build()

    model = FaceTracker(facetracker)
    model.compile(opt,classloss, regressloss, metrics=['accuracy'])

    logdir='logs'
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    hist = model.fit(train, epochs=epoch, validation_data=val, callbacks=[tensorboard_callback])

    if(save_model):facetracker.save('facedetector.h5')


def build(): 
        input_layer = Input(shape=(120,120,3))
        
        base_model = InceptionV3(include_top=False,weights = "imagenet")(input_layer)

        f1 = GlobalMaxPooling2D()(base_model)
        class1 = Dense(2048, activation='relu')(f1)
        class2 = Dense(1, activation='sigmoid')(class1)
   
        f2 = GlobalMaxPooling2D()(base_model)
        f2 = Dropout(0.5)(f2)
        regress1 = Dense(2048, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.1, l2=0.01))(f2)
        regress1 = Dropout(0.5)(regress1)
        regress2 = Dense(4, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.1, l2=0.01))(regress1)
        
        tracker = Model(inputs=input_layer, outputs=[class2, regress2])
        return tracker

def localization_loss(y_true, yhat):            
    delta_coord = tf.reduce_sum(tf.square(y_true[:,:2] - yhat[:,:2]))
                    
    h_true = y_true[:,3] - y_true[:,1] 
    w_true = y_true[:,2] - y_true[:,0] 

    h_pred = yhat[:,3] - yhat[:,1] 
    w_pred = yhat[:,2] - yhat[:,0] 
        
    delta_size = tf.reduce_sum(tf.square(w_true - w_pred) + tf.square(h_true-h_pred))
        
    return delta_coord + delta_size

