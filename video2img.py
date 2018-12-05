import cv2
import os
import time

# code from https://www.youtube.com/watch?v=uL-wCzVMPsc
nb_frame_per_video = 100
cap = cv2.VideoCapture('D:/Users/alexa_000/Documents/5GE/TDSI/Proyectos/Challenges/data/15-gare_venissieux/vlc-record-2018-10-24-23h05m26s-VID_20181013_152350.mp4-.mp4')
video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
print('video_length : ' + str(video_length))
time_start = time.time()  # Log the time
print('time_start : ' + str(time_start))
framePos = cap.get(cv2.CAP_PROP_POS_FRAMES)  # frame position
frameRate = int(video_length/nb_frame_per_video)
# https://docs.opencv.org/3.1.0/d8/dfe/classcv_1_1VideoCapture.html#aeb1644641842e6b104f244f049648f94
print('framePos : ' + str(framePos))
print('frameRate : ' + str(frameRate))
try:
    if not os.path.exists('data'):
        os.makedirs('data')
except OSError:
    print('Error: Creating directory of data')

currentFrame = 0
nb_frame = 0
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    framePos = cap.get(cv2.CAP_PROP_POS_FRAMES)  # frame position

    if framePos % frameRate == 0:
        print('framePos : ' + str(framePos))
        # Saves image of the current frame in jpg file each 'frameRate' frame
        name = 'D:/Users/alexa_000/Documents/5GE/TDSI/Proyectos/Challenges/data/15-gare_venissieux/data_out/15-gare_venissieux' + str(nb_frame) + '.jpg'
        nb_frame += 1
        print('Creating... ' + name)
        frame_scaled = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
        cv2.imwrite(name, frame_scaled)

    # To stop duplicate images
    currentFrame += 1
    if currentFrame == video_length:
        break

# Log the time again
time_end = time.time()
print('time_end : ' + str(time_end))
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()  