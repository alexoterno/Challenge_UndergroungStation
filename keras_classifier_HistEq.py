import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
import cv2
import math
from keras.utils.np_utils import to_categorical
from keras.utils import plot_model
import json
from PIL import Image

# size of our images.
img = cv2.imread('/my_data/data_train/00_einstein/00_einstein0.jpg')  # an test image just to get the size
img_height, img_width, channels = img.shape
print('height : ' + str(img_height)+' width : ' + str(img_width))
# img_width, img_height = 224, 224

top_model_weights_path = 'bottleneck_fc_model.h5'
model_architecture = 'model_architecture.json'
train_data_dir = '/my_data/data_train'
validation_data_dir = '/my_data/data_validation'
nb_train_samples = 1349
nb_validation_samples = 234
epochs = 50  # One Epoch is when an ENTIRE dataset is passed forward and backward through the neural network only ONCE
batch_size = 16  # Total number of training examples present in a single batch
number_class = 26

# EQUALIZE HISTOGRAM
def imhist(im):
  # calculates normalized histogram of an image
	m, n = im.shape
	h = [0.0] * 256
	for i in range(m):
		for j in range(n):
			h[int(im[i, j])]+=1
	return np.array(h)/(m*n)

def cumsum(h):
	# finds cumulative sum of a numpy array, list
	return [sum(h[:i+1]) for i in range(len(h))]

def histeq(im):
	#calculate Histogram
	h = imhist(im)
	cdf = np.array(cumsum(h)) #cumulative distribution function
	sk = np.uint8(255 * cdf) #finding transfer function values
	s1, s2 = im.shape
	Y = np.zeros_like(im)
	# applying transfered values for each pixels
	for i in range(0, s1):
		for j in range(0, s2):
			Y[i, j] = sk[int(im[i, j])]
	H = imhist(Y)
	#return transformed image, original and new istogram,
	# and transform function
	return Y

def myFunc(image):

    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    H,S,V = cv2.split(lab_image)
    V_eq = histeq(V)
    img_lab_eq = cv2.merge((H, S, V_eq))
    img_lab_eq = cv2.cvtColor(lab_image, cv2.COLOR_HSV2RGB)
    return img_lab_eq


def save_bottlebeck_features():
    # Generate batches of tensor image data with real-time data augmentation
    datagen = ImageDataGenerator(rescale=1. / 255, preprocessing_function=myFunc)  # value 0-255 to 0-1

    # build the VGG16 network (none including the 3 fully-connected layers at the top of the network)
    # 'imagenet' (pre-training on ImageNet)
    model = applications.VGG16(include_top=False, weights='imagenet', classes=number_class)

    # Takes the path to a directory & generates batches of augmented data
    train_generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    predict_size_train = int(math.ceil(nb_train_samples / batch_size))
    # Generates predictions for the input samples from a data generator
    # save_to_dir, save_prefix, save_format, to DO ?
    bottleneck_features_train = model.predict_generator(train_generator, predict_size_train)

    # Save an array to a binary file in NumPy .npy format
    np.save(open('bottleneck_features_train.npy', 'wb'),
            bottleneck_features_train)
    train_labels = train_generator.classes
    # convert the training labels to categorical vectors (Converts a class vector (integers) to binary class matrix)
    train_labels = to_categorical(train_labels, num_classes=number_class)
    np.save(open('train_labels.npy', 'wb'), train_labels)

    valid_generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    predict_size_validation = int(math.ceil(nb_validation_samples / batch_size))
    bottleneck_features_validation = model.predict_generator(valid_generator, predict_size_validation)
    np.save(open('bottleneck_features_validation.npy', 'wb'),
            bottleneck_features_validation)
    validation_labels = valid_generator.classes
    validation_labels = to_categorical(validation_labels, num_classes=number_class)
    np.save(open('validation_labels.npy', 'wb'), validation_labels)

    # get the class_dictionary, i.e each number is associated to a station name and save in .json file
    class_dictionary = valid_generator.class_indices
    print("class_dictionary : " + str(class_dictionary))
    with open('class_dictionary.json', 'w') as fp:
        json.dump(class_dictionary, fp, sort_keys=False, indent=4)


def train_top_model():
    # load the train_data and train_labels
    train_data = np.load(open('bottleneck_features_train.npy', 'rb'))  # rb : files that don’t contain text
    # (read and written in the form of bytes objects) Normally, files are opened in text mode
    train_labels = np.load(open('train_labels.npy', 'rb'))

    # load the validation_data and validation_labels
    validation_data = np.load(open('bottleneck_features_validation.npy', 'rb'))
    validation_labels = np.load(open('validation_labels.npy', 'rb'))

    # build top model
    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(number_class, activation='softmax'))

    # compile the model to configure the learning process for a multi-class classification problem
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    # train the model (Keras models are trained on Numpy arrays of input data and labels)
    model.fit(train_data, train_labels,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(validation_data, validation_labels))

    # save the weights of our trained model
    model.save_weights(top_model_weights_path)

    # create and save the architecture of our model into a .json file
    with open(model_architecture, 'w') as f:
        f.write(model.to_json())

    # prints a summary representation of your model
    model.summary()

    # Converts a Keras model to dot format and save to a file
    plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True, rankdir='TB')


save_bottlebeck_features()
train_top_model()