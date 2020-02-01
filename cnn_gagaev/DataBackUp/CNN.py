## Proj 2, CNN, Gagaev
from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
import matplotlib.pylab as plt
from keras.utils import plot_model
from matplotlib import pyplot
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img
import numpy as np


batch_size = 128
num_classes = 10
epochs = 10

# input image dimensions
img_x, img_y = 28, 28

# load the MNIST data set, which already splits into train and test sets for us
(x_train, y_train), (x_test, y_test) = mnist.load_data()


datagen = ImageDataGenerator()
# prepare an iterators for each dataset
train_it = datagen.flow_from_directory('train/', class_mode='binary',target_size=(28,28),batch_size=3949,shuffle="false")
test_it = datagen.flow_from_directory('test/', class_mode='binary',target_size=(28,28),batch_size=618,shuffle="false")

# confirm the iterator works
x_train, y_train = train_it.next()
x_test, y_test = test_it.next()

# reshape the data into a 4D tensor - (sample_number, x_img_size, y_img_size, num_channels)
x_train = x_train.reshape(x_train.shape[0], img_x, img_y, 3)
x_test = x_test.reshape(x_test.shape[0], img_x, img_y, 3)
input_shape = (img_x, img_y, 3)

# convert the data to the right type
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Building neural network
model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])




history =  model.fit(x_train, keras.utils.to_categorical(y_train, num_classes),
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, keras.utils.to_categorical(y_test, num_classes)))



model.save_weights('model.h5')
#saving the graph representation of network structure
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True) 

predictions = model.predict(x_test[:])
PredArr = np.argmax(predictions, axis=1)
print(PredArr) # [7, 2, 1, 0, 4]
y_test = y_test.astype(int)  
print(y_test[:]) # [7, 2, 1, 0, 4]



with open('prediction_comparison.txt', 'w') as f:
    for i in range(len(y_test)):
        f.write(str(y_test[i].astype(int)))  
        f.write('\t | \t')
        f.write(str(PredArr[i]))
        f.write('\n')


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('accuracy_graph.png')
plt.show()

#creating the loss plot
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('loss_graph.png')
plt.show()

#Due to a reason that i have a lot of images, I pick one image and try to 
#classify it and try to look what is happening if we have no labels
from keras.preprocessing import image
def test_single_image(path):
  animals = ['LegoBrick corner','LegoBrick 2x2', 'LegoBrick 1x2',
             'LegoBrick 1x1', 'LegoPlate 2x2', 'LegoPlate 1x2', 
             'LegoPlate 1x1', 'RoofTile 1x2x45', 'Plate 1x2 with knob']
  images = image.load_img('One_image/4/0001.png', target_size = (28, 28))
  images.show()
  images = image.img_to_array(images)
  images = np.expand_dims(images, axis = 0)
  bt_prediction = model.predict(images) 
  #  preds = model.predict_proba(bt_prediction)
  for idx, animal, x in zip(range(0,9), animals , bt_prediction[0]):
   print("ID: {}, Label: {} {}%".format(idx, animal, round(x*100,2) ))
  print('Final Decision:')
  for x in range(3):
   print('.'*(x+1))
  class_predicted = model.predict_classes(images)
  class_dictionary = train_it.class_indices 
  inv_map = {v: k for k, v in class_dictionary.items()} 
  print("ID: {}, Label: {}".format(class_predicted[0],  inv_map[class_predicted[0]])) 
  return image.load_img(path)
path = 'One_image/5/0001.png'
test_single_image(path)
