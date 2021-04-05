import os

import tensorflow as tf
import keras
import argparse
from myconfig import configt
from keras import optimizers
from keras.utils import np_utils
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))

#conv_base = ResNet50(weights='imagenet', include_top=False, input_shape=(230, 230, 3))
#from keras.applications.xception import Xception
#conv_base = Xception(weights='imagenet', include_top=False, input_shape=(96, 96, 3))


from keras import models
from keras import layers
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import numpy as np
from imutils import paths
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.layers import Average
from keras.layers import merge

from keras import applications
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.models import Model, Input

import argparse

from keras import optimizers
import seaborn as sns
#from keras.engine.topology import Input

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

# initialize our number of epochs, initial learning rate, and batch
# size


chanDim=-1

def resnet500(model_input):
    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=model_input)
    last = base_model.output
    x = layers.Flatten()(last)
    x = Dense(256, activation='relu')(x)
    #preds = Dense(8, activation='softmax')(x)
    #model = Model(base_model.input, preds)
    model = Model(base_model.input, x)

    base_model.trainable = True
    set_trainable = False
    for layer in base_model.layers:
        if layer.name == 'add_30': # 从这一层开始往后均可训练
            set_trainable = True
            if set_trainable:
                layer.trainable = True
            else:
                layer.trainable = False
    
    return model

def vgg166(model_input):
    base_model = VGG16(weights='imagenet', include_top=False, input_tensor=model_input)
    last = base_model.output
    x = layers.Flatten()(last)
    x = Dense(256, activation='relu')(x)
    #preds = Dense(8, activation='softmax')(x)
    model = Model(base_model.input, x)

    base_model.trainable = True
    set_trainable = False
    for layer in base_model.layers:
        if layer.name == 'block5_conv1': # 从这一层开始往后均可训练
            set_trainable = True
            if set_trainable:
                layer.trainable = True
            else:
                layer.trainable = False
    
    return model

model_input = Input(shape=(230, 230, 3))


model1 = resnet500(model_input)
model2 = vgg166(model_input)

ensembled_models = [model1,model2]
def ensemble(models,model_input):
    outputs = [model.outputs[0] for model in models]
    #modelo1 = model1.outputs[0]
    #modelo2 = model2.outputs[0]
    modelo1 = outputs[0]
    modelo2 = outputs[1]
    y = layers.concatenate([modelo1, modelo2], axis=-1)
    preds = Dense(8, activation='softmax')(y)
    model = Model(model_input,preds,name='ensemble')
    return model


ensemble_model = ensemble(ensembled_models,model_input)


from keras.utils import plot_model
plot_model(ensemble_model, show_shapes=True, to_file='bcdnet_comodel.png')
#added = keras.layers.add([input1, input2])


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

# initialize our number of epochs, initial learning rate, and batch
# size
NUM_EPOCHS = 38
INIT_LR = 1e-2
BS = 16

# determine the total number of image paths in training, validation,
# and testing directories
trainPaths = list(paths.list_images(configt.TRAIN_PATH))
totalTrain = len(trainPaths)
totalVal = len(list(paths.list_images(configt.VAL_PATH)))
totalTest = len(list(paths.list_images(configt.TEST_PATH)))

# account for skew in the labeled data
trainLabels = [int(p.split(os.path.sep)[-2]) for p in trainPaths]
trainLabels = np_utils.to_categorical(trainLabels)
classTotals = trainLabels.sum(axis=0)
classWeight = classTotals.max() / classTotals

# initialize the training training data augmentation object
trainAug = ImageDataGenerator(
	rescale=1 / 255.0,
	rotation_range=20,
	zoom_range=0.05,
	width_shift_range=0.1,
	height_shift_range=0.1,
	shear_range=0.05,
	horizontal_flip=True,
	vertical_flip=True,
	fill_mode="nearest")

# initialize the validation (and testing) data augmentation object
valAug = ImageDataGenerator(rescale=1 / 255.0)

# initialize the training generator
trainGen = trainAug.flow_from_directory(
	configt.TRAIN_PATH,
	class_mode="categorical",
	target_size=(230, 230),
	color_mode="rgb",
	shuffle=True,
	batch_size=BS)

# initialize the validation generator
valGen = valAug.flow_from_directory(
	configt.VAL_PATH,
	class_mode="categorical",
	target_size=(230, 230),
	color_mode="rgb",
	shuffle=False,
	batch_size=BS )

# initialize the testing generator
testGen = valAug.flow_from_directory(
	configt.TEST_PATH,
	class_mode="categorical",
	target_size=(230, 230),
	color_mode="rgb",
	shuffle=False,
	batch_size=BS)

# initialize our CancerNet model and compile it

#opt = Adagrad(lr=INIT_LR, decay=INIT_LR / NUM_EPOCHS)
#opt= Ranger(params=CancerNet.parameters(), lr=1e-3, alpha=0.5, k=6, N_sma_threshhold=5, betas=(.95,0.999), eps=1e-5, weight_decay=0)
#opt = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#opt = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=0.5, decay=INIT_LR / NUM_EPOCHS)
#opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
#opt = optimizers.Adadelta(lr=0.01, rho=0.95, epsilon=0.05, decay=1e-2)
#opt = optimizers.Adamax(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
opt = optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
ensemble_model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])



# 绘制训练过程中的损失曲线和精度曲线




H = ensemble_model.fit_generator(
	trainGen,
	steps_per_epoch=totalTrain // BS,
	validation_data=valGen,
	validation_steps=totalVal // BS,
	class_weight=classWeight,
	epochs=NUM_EPOCHS)

# reset the testing generator and then use our trained model to
# make predictions on the data
print("[INFO] evaluating network...")
testGen.reset()
predIdxs = ensemble_model.predict_generator(testGen,
	steps=(totalTest // BS) + 1)

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
# label with corresponding largest predicted probability
ensemble_model.save("zuhe.h5")
predIdxs = np.argmax(predIdxs, axis=1)
print(confusion_matrix(testGen.classes, predIdxs))
# show a nicely formatted classification report
print(classification_report(testGen.classes, predIdxs,
	target_names=testGen.class_indices.keys(), digits=4))

# compute the confusion matrix and and use it to derive the raw
# accuracy, sensitivity, and specificity


import matplotlib.pyplot as plt

acc = H.history['acc']
val_acc = H.history['val_acc']
loss = H.history['loss']
val_loss = H.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'b:', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.plot(epochs, loss, 'b--', label='Training loss')
plt.plot(epochs, val_loss, 'r-.', label='Validation loss')
plt.title('Training Loss and Accuracy on Dataset')
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()




# === 混淆矩阵：真实值与预测值的对比 ===
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
con_mat = confusion_matrix(testGen.classes, predIdxs)

con_mat_norm = con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis]     # 归一化

con_mat_norm = np.around(con_mat_norm, decimals=4)

# === plot ===
plt.figure(figsize=(8, 8))
sns.heatmap(con_mat_norm, annot=True, cmap='Blues')

plt.ylim(0, 8)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')


plt.show()





