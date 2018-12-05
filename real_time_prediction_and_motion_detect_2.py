import numpy as np
from keras.preprocessing.image import img_to_array
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras import applications
import cv2
import json
import os

# force to use CPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

threshold_rate = 0.2  # we display only the station name which have a certainty > threshold_rate
value = 0.0  # certainty of the station name
top_model_weights_path = "D:/Users/alexa_000/Documents/5GE/TDSI/Proyectos/Challenges/bottleneck_fc_model/" \
                         "bottleneck_fc_model.h5"  # weights path of the trained model
num_classes = 15
count = 0  # count the number of frame
count2 = 1  # count the number of prediction
analysed_frame_rate = 15  # each "analysed_frame_rate" frame we do a prediction

with open('D:/Users/alexa_000/Documents/5GE/TDSI/Proyectos/Challenges/class_dictionary/class_dictionary.json', 'r') \
        as fp:
    class_dictionary = json.load(fp)  # class dictionary which gives the station name associated with a number
print("class_dictionary : " + str(class_dictionary))

label = '...'  # the station name
cap = cv2.VideoCapture('D:/Users/alexa_000/Documents/5GE/TDSI/Proyectos/Challenges/videos_6-11/'
                       '03-gorgeloup2.mp4')  # open the test video

sdThresh = 13  # motion detect threshold
font = cv2.FONT_HERSHEY_SIMPLEX  # font


def dist_map(frame11, frame22):
    """outputs pythagorean distance between two frames"""
    frame11_32 = np.float32(frame11)
    frame22_32 = np.float32(frame22)
    diff32 = frame11_32 - frame22_32
    norm32 = np.sqrt(diff32[:, :, 0]**2 + diff32[:, :, 1]**2 + diff32[:, :, 2]**2)/np.sqrt(255**2 + 255**2 + 255**2)
    distance = np.uint8(norm32*255)
    return distance


cv2.namedWindow('frame')
# cv2.namedWindow('dist')
# cv2.moveWindow('frame', 20, 150) # to modify the position window en the screen
_, frame1 = cap.read()  # capture the 1st frame
_, frame2 = cap.read()  # capture the second frame
img_height, img_width, channels = frame1.shape  # image size
# print('height : ' + str(img_height)+' width : ' + str(img_width))
img_width = int(img_width/2)
img_height = int(img_height/2)
# print('height/2 : ' + str(img_height)+' width/2 : ' + str(img_width))
frame1 = cv2.transpose(frame1)
frame1 = cv2.flip(frame1, 1)  # transpose + flip = 90° rotation, because we grab the videos in a landscape format
frame2 = cv2.transpose(frame2)
frame2 = cv2.flip(frame2, 1)

# create a .txt file to save some quantifiable data
file = open('D:/Users/alexa_000/Documents/5GE/TDSI/Proyectos/Challenges/videos_6-11/03-gorgeloup2.txt', 'w')


while True:
    _, frame3 = cap.read()  # capture the 3rd frame
    # rows, cols, _ = np.shape(frame3)
    # cv2.imshow('dist', frame3)
    frame3 = cv2.transpose(frame3)
    frame3 = cv2.flip(frame3, 1)
    dist = dist_map(frame1, frame3)  # pythagorean distance the 1st and the 3rd frame

    frame1 = frame2
    frame2 = frame3

    # apply Gaussian smoothing
    mod = cv2.GaussianBlur(dist, (5, 5), 0)

    cv2.putText(frame2, "Predicted : {}".format(label) + " Rate : " + str(value), (10, 30), font, 0.9, (43, 99, 255), 2,
                cv2.LINE_AA)
    # file.write("Predicted : {}".format(label) + " Rate : " + str(value))

    # calculate st dev test
    _, stDev = cv2.meanStdDev(mod)

    # motion detect condition
    if stDev > sdThresh:
        cv2.putText(frame2, "Standard Deviation - {}".format(round(stDev[0][0], 0)),
                    (10, 65), font, 1.1, (43, 99, 255), 2, cv2.LINE_AA)
        # file.write("Standard Deviation - {}".format(round(stDev[0][0], 0)))

    if stDev < sdThresh:
        cv2.putText(frame2, "Standard Deviation - {}".format(round(stDev[0][0], 0)) + " The subway is stopped",
                    (10, 65), font, 0.9, (43, 99, 255), 2, cv2.LINE_AA)
        file.write("Standard Deviation - {}".format(round(stDev[0][0], 0)) + " The subway is stopped\n")

    frame2_resize = cv2.resize(frame2, (img_height, img_width), interpolation=cv2.INTER_AREA)  # resize the frame
    cv2.imshow('frame', frame2_resize)  # display the frame

    # to check if the user pressed the character 'q': to quite the reading video
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    count += 1

    # prediction condition
    """ We first run the image through the pre - trained VGG16 model(without the fully - connected layers again) and get 
    the bottleneck predictions. We then run the bottleneck prediction through the trained top model - which we created 
    in the previous step - and get the final classification."""
    if count % analysed_frame_rate == 0:
        img_orig = cv2.resize(frame3, (224, 224), interpolation=cv2.INTER_AREA)
        img = img_to_array(img_orig)

        # important! otherwise the predictions will be '0'
        img = img / 255
        img = np.expand_dims(img, axis=0)

        # build the VGG16 network
        model = applications.VGG16(include_top=False, weights='imagenet')

        # get the bottleneck prediction from the pre-trained VGG16 model
        bottleneck_prediction = model.predict(img)

        # build top model
        model = Sequential()
        model.add(Flatten(input_shape=bottleneck_prediction.shape[1:]))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))

        # load the weights of the trained model
        model.load_weights(top_model_weights_path)

        # use the bottleneck prediction on the top model to get the final classification
        class_predicted = model.predict(bottleneck_prediction)
        # print(class_predicted)

        # we take, write and display only the station name which have a certainty > threshold_rate
        values = class_predicted[class_predicted > threshold_rate]
        file.write("Values bigger than" + str(threshold_rate) + "  = " + str(values) + "\n")
        print("Values bigger than " + str(threshold_rate) + " = " + str(values))
        # take the index of value which have a certainty > threshold_rate
        x = np.argwhere(class_predicted > threshold_rate)
        # print(x)
        # we keep the second col of x
        x1 = x[:, 1]
        # print(x1)
        # take the item of class_dictionzry
        inv_map = {v: k for k, v in class_dictionary.items()}

        # to display the station name which have certainty > threshold_rate
        for inID in x1:
            label = inv_map[inID]
            # get the prediction label
            file.write(str(count2) + " Image ID: {}, Label: {}".format(inID, label) + "\n")
            print(str(count2) + " Image ID: {}, Label: {}".format(inID, label))

        value = np.amax(values)
        count2 += 1

# end
file.close()
cap.release()
cv2.destroyAllWindows()
