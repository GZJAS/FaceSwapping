import numpy as np
import cv2
from facial_landmarks import *
from estimateH import *
import matplotlib.pyplot as plt

im1 = cv2.imread("obama.jpg", cv2.IMREAD_COLOR)
im2 = cv2.imread("trump.jpg", cv2.IMREAD_COLOR)

try:
	src_pts = get_landmarks(im1)
	dst_pts = get_landmarks(im2)
except NameError:
	print "Error"

H = estimate_H(src_pts, dst_pts)
H = H[0]

# Show the images
plt.imshow(im1[..., ::-1])
plt.scatter(np.array(src_pts)[:,0],np.array(src_pts)[:,1], 0.3)
plt.show(block=False)

plt.figure()
plt.imshow(im2[..., ::-1])
plt.scatter(np.array(dst_pts)[:,0],np.array(dst_pts)[:,1], 0.3)
plt.show()

