import matplotlib.pyplot as plt
import numpy as np
import cv2

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
    # H,S,V = cv2.split(lab_image)
    # V_eq = histeq(V)
    # img_lab_eq = cv2.merge((H, S, V_eq))
    img_lab_eq = cv2.cvtColor(lab_image, cv2.COLOR_HSV2RGB)
    return img_lab_eq


cv2.destroyAllWindows()

# size of our image
img_day = cv2.imread('C:/Users/isabe/OneDrive/Documentos/INSA-2014-2019/2018-2019/TDSI/PTI/Datasets/09_servient/258.jpg')  # a test image just to get the size
img_night = cv2.imread('C:/Users/isabe/OneDrive/Documentos/INSA-2014-2019/2018-2019/TDSI/PTI/Datasets/09_servient/23.jpg')  # a test image just to get the size

cv2.imshow('day',img_day)
cv2.imshow('night',img_night)

brightLAB = cv2.cvtColor(img_day, cv2.COLOR_RGB2HSV)
darkLAB = cv2.cvtColor(img_night, cv2.COLOR_RGB2HSV)

dH,dS,dV = cv2.split(brightLAB)
nH,nS,nV = cv2.split(darkLAB)


"""
# CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
dL_clahe = clahe.apply(dL)
nL_clahe = clahe.apply(nL)
night_clahe = cv2.merge((nL_clahe,na,nb))
day_clahe = cv2.merge((dL_clahe,da,db))
night_clahe = cv2.cvtColor(night_clahe, cv2.COLOR_LAB2RGB)
day_clahe = cv2.cvtColor(day_clahe, cv2.COLOR_LAB2RGB)
cv2.imshow('night_clahe',night_clahe)
cv2.imshow('day_clahe',day_clahe)
"""
# Normal histogramme equalization
#nV_eq = cv2.equalizeHist(nV)
#dV_eq = cv2.equalizeHist(dV)

nV_eq=histeq(nV)
dV_eq=histeq(dV)


# Merge Lab

night_hist_eq = cv2.merge((nH,nS,nV_eq))
day_hist_eq = cv2.merge((dH,dS,dV_eq))

# Lab to BGR
night_hist_eq = cv2.cvtColor(night_hist_eq, cv2.COLOR_HSV2RGB)
day_hist_eq = cv2.cvtColor(day_hist_eq, cv2.COLOR_HSV2RGB)
cv2.imshow('night_histeq',night_hist_eq)
cv2.imshow('day_histeq',day_hist_eq)


"""
"""
# Calcul des histogrammes
ys_day = cv2.calcHist([dV], [0], None, [256], [0,256])
ys_night = cv2.calcHist([nV], [0], None, [256], [0,256])
ys_day_eq = cv2.calcHist([dV_eq], [0], None, [256], [0,256])
ys_night_eq = cv2.calcHist([nV_eq], [0], None, [256], [0,256])



# Plot histogrammes
fig = plt.figure(figsize=(16,8))
xs= np.arange(256)

ax = fig.add_subplot(321)
ax.plot(xs, ys_day.ravel())
ax = fig.add_subplot(322)
ax.plot(xs,ys_night.ravel())

ax = fig.add_subplot(323)
ax.plot(xs, ys_day_eq.ravel())
ax = fig.add_subplot(324)
ax.plot(xs,ys_night_eq.ravel())


ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.show()



cv2.waitKey(0)
cv2.destroyAllWindows()


