#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import os, random
import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from matplotlib.image import imread
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, Activation
from tqdm import tqdm

tf.keras.backend.clear_session()

# Select a particular GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Increase GPU memory gradually
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

# Define Batch size and number of Epochs
BATCH_SIZE = 100
EPOCHS = 95

# Image paths for training and test
train_x_files = sorted(os.listdir(os.getcwd()+'/dataset/training_set'),key=lambda x: int(x.split(".")[1]))
test_x_files = sorted(os.listdir(os.getcwd()+'/dataset/test_set'),key=lambda x: int(x.split(".")[1]))
train_folder = os.getcwd()+"/dataset/training_set/"
test_folder = os.getcwd()+"/dataset/test_set/"

# Parameters for Tensor Board
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

# Lists for storing accuracy and loss values
train_accuracy_list = []
test_accuracy_list = []
train_loss_list = []
test_loss_list = []

# Early Stopping Parameters
patience_cnt = 0
patience = 5
best_loss = 1.0

################################################### Custom Data Generator Class #############################################
def Data_Generator(folder, files_batch):
  for filename in files_batch:
    image_path = folder + filename
    image = imread(image_path)
    label = image_path.split(os.path.sep)[-1]
    if ('stego' in label):
      label = 1
    else:
      label = 0
    image = np.array(image, dtype='float') / 255.0
    image = np.expand_dims(image, axis = 2)
    label = np.array(label, dtype='float')
    label = to_categorical(label, num_classes = 2)
    yield image, label


################################################# Tensorflow Model ##########################################################
# Refer "Steganalysis via CNN using large convolution filters for embedding process 
#        with same stego key: A deep learning approach for telemedicine" Paper for model architecture
class Conv_Model(Model):
    def __init__(self):
        super(Conv_Model, self).__init__()
        self.conv1 = Conv2D(1, kernel_size = 3, input_shape = (512, 512, 1))
        self.act1 = Activation('tanh')
        self.conv2 = Conv2D(64, kernel_size = 509)
        self.act2 = Activation('tanh')
        self.flatten = Flatten()
        self.dense2 = Dense(2)
        self.act4 = Activation('softmax')
    
    def call(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.flatten(x)
        x = self.dense2(x)
        return self.act4(x)

# Create an instance of Tensorflow Model
classifier = Conv_Model()

# Choose optimizer and loss function for training
loss_object = tf.keras.losses.CategoricalCrossentropy()
initial_learning_rate = 0.005
decay_steps = 1.0
decay_rate = 0.0000005
learning_rate_function = tf.keras.optimizers.schedules.InverseTimeDecay(initial_learning_rate = initial_learning_rate, 
                                                                        decay_steps = decay_steps, 
                                                                        decay_rate = decay_rate, 
                                                                        staircase = True)
optimizer = tf.keras.optimizers.SGD(learning_rate = learning_rate_function, momentum = 0.0, nesterov = False)

# Select metrics to measure the loss and the accuracy of the model. These metrics accumulate the values over 
# epochs and then print the overall result.
train_loss = tf.keras.metrics.Mean(name = 'train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name = 'train_accuracy')

test_loss = tf.keras.metrics.Mean(name = 'test_loss')
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name = 'test_accuracy')

# Use tf.GradientTape to train the model
@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = classifier(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, classifier.trainable_variables)
    optimizer.apply_gradients(zip(gradients, classifier.trainable_variables))
    
    train_loss(loss)
    train_accuracy(labels, predictions)

# Test the model
@tf.function
def test_step(images, labels):
    predictions = classifier(images)
    t_loss = loss_object(labels, predictions)
    
    test_loss(t_loss)
    test_accuracy(labels, predictions)


# Generate test dataset from custom generator
test_dataset = tf.data.Dataset.from_generator(lambda : Data_Generator(test_folder, test_x_files), (tf.float32, tf.float32))
test_dataset = test_dataset.batch(batch_size = BATCH_SIZE)


########################### Start of training and testing the model for given number of epochs ##########################
for epoch in range(EPOCHS):
  # Generate training dataset from custom generator in each epoch
  train_dataset = tf.data.Dataset.from_generator(lambda : Data_Generator(train_folder, train_x_files), (tf.float32, tf.float32))
  train_dataset = train_dataset.batch(batch_size = BATCH_SIZE)
  train_dataset = train_dataset.shuffle(BATCH_SIZE)
  
  # Training batchwise uisng training dataset
  for train_batch, (train_images, train_labels) in tqdm(enumerate(train_dataset), total = len(train_x_files)/BATCH_SIZE):
    train_step(train_images, train_labels)
  
  # Store the metrics to visualize in Tensor Board  
  with train_summary_writer.as_default():
      tf.summary.scalar('loss', train_loss.result(), step = epoch)
      tf.summary.scalar('accuracy', train_accuracy.result(), step = epoch)
  
  # Testing batchwise uisng test dataset
  for test_batch, (test_images, test_labels) in tqdm(enumerate(test_dataset), total = len(test_x_files)/BATCH_SIZE):
    test_step(test_images, test_labels)

  # Store the metrics to visualize in Tensor Board  
  with test_summary_writer.as_default():
      tf.summary.scalar('loss', test_loss.result(), step = epoch)
      tf.summary.scalar('accuracy', test_accuracy.result(), step = epoch)
  
  # Store the metrics in their respective lists
  train_accuracy_list.append(train_accuracy.result())
  test_accuracy_list.append(test_accuracy.result())
  train_loss_list.append(train_loss.result())
  test_loss_list.append(test_loss.result())
  
  # Display the values after each epoch end
  print('\n')
  template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
  print(template.format(epoch+1, 
                        train_loss.result(), 
                        train_accuracy.result()*100,
                        test_loss.result(), 
                        test_accuracy.result()*100))
  print('==================================================================================================================')
  
  # Reset the metrics for the next epoch
  train_loss.reset_states()
  train_accuracy.reset_states()
  test_loss.reset_states()
  test_accuracy.reset_states()
  
  # Code for Early Stopping
  if (epoch > 0):
      if(test_loss_list[epoch] < best_loss):
          best_loss = test_loss_list[epoch];
          patience_cnt = 0
      else:
          patience_cnt = patience_cnt + 1
      
      if (patience_cnt > patience):
          print("Stopping training of model due to Early Stopping")
          break
######################################## End of Training and Testing process ###############################################

############################################ Plots for Accuracy and Loss ######################################################
# Accuracy Plot
plt.plot(train_accuracy_list)
plt.plot(test_accuracy_list)
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Number of epochs')
plt.legend(['Training', 'Testing'], loc='lower right')
plt.grid()
plt.show()
plt.savefig('results_1/accuracy.png')

# Loss Plot
plt.plot(train_loss_list)
plt.plot(test_loss_list)
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Number of epochs')
plt.legend(['Training', 'Testing'], loc='upper right')
plt.grid()
plt.show()
plt.savefig('results_1/loss.png')

################################################ Save the Trained Model #####################################################
tf.saved_model.save(classifier, 'Packetloss_vs_WOW_1_Tensorflow_Model')

################################################### Calculate Detection Accuracy ############################################
from sklearn.metrics import confusion_matrix
classifier = tf.saved_model.load('Packetloss_vs_WOW_1_Tensorflow_Model')
predict_x_files = sorted(os.listdir(os.getcwd()+'/dataset/prediction_set'),key=lambda x: int(x.split(".")[1]))
predict_folder = os.getcwd()+"/dataset/prediction_set/"

predict_images = []
predict_labels = []
# Predict for the first 4000 images from the prediction_set folder
for filename in range(4000):
    image_path = predict_folder + predict_x_files[filename]
    image = imread(image_path)
    label = image_path.split(os.path.sep)[-1]
    if ('stego' in label):
      label = 1
    else:
      label = 0
    predict_images.append(image)
    predict_labels.append(label)
   
predict_images = np.array(predict_images, dtype='float32') / 255.0
predict_images = np.expand_dims(predict_images, axis = 3)

predictions = classifier(predict_images)

y_pred = predictions > 0.5
y_pred = np.argmax(y_pred, axis = 1)
conf_matrix = confusion_matrix(predict_labels, y_pred)

############################################### Write metrics list into files for future reference ############################
import pickle
with open('training_accuracy.txt', 'wb') as f:
    pickle.dump(train_accuracy_list, f)

with open('training_loss.txt', 'wb') as f:
    pickle.dump(train_loss_list, f)
    
with open('test_accuracy.txt', 'wb') as f:
    pickle.dump(test_accuracy_list, f)

with open('test_loss.txt', 'wb') as f:
    pickle.dump(test_loss_list, f)  
    
# with open('results_1/test_accuracy.txt', 'rb') as f:
#     list_acc = pickle.load(f)

################################################ Plot Confusion Matrix #######################################################
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'
            
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

plot_confusion_matrix(conf_matrix, classes=['Cover', 'Stego'],
                      title='Confusion matrix')