from random import shuffle
import os
import cv2
path = 'D:/Users/alexa_000/Documents/5GE/TDSI/Proyectos/Challenges/dataset_5/data_out/'
files = [os.path.join(path, fle) for fle in os.listdir(path) if fle.endswith(".jpg")]
shuffle(files)
print(files)
print(files[0])
count = 0
for k in files:
    print(k)
    image = cv2.imread(k)
    # if you need to rotate the frame
    # image = cv2.transpose(image)
    # image = cv2.flip(image, 1)
    cv2.imwrite('D:/Users/alexa_000/Documents/5GE/TDSI/Proyectos/Challenges/dataset_5/'
                '01-garevaise/01-garevaise'
                + '%d.jpg' % count, image)
    count += 1