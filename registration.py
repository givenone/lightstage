import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

input_path = "./Front/"
input_names = ["x", "x_c", "y", "y_c", "z", "z_c", "w"]
input_format = ".bmp"

ref_path = "./Front/"
ref_name = "1"
ref_format = ".bmp"

mov_path = "./Front/"
mov_name = "w"
mov_format = ".bmp"


output_path = input_path + 'xformed_a/'

if not os.path.exists(output_path):
    os.mkdir(output_path)

ref = ref_path + ref_name + ref_format
#ref = io.imread(ref_name,as_gray=True)

mov = mov_path + mov_name + mov_format
#mov = (mov / (np.mean(mov) / np.mean(ref))).astype("uint8")

# Open the image files. 
img1_color = cv2.imread(mov)  # Image to be aligned. 
img2 = cv2.imread(ref, cv2.IMREAD_GRAYSCALE)    # Reference image. 
#img2_blur = cv2.medianBlur(img2, 5)  
img2 = np.clip((img2 * 1.3), 0, 255).astype('uint8')
# Convert to grayscale. 
img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY) 
#img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY) 
height, width = img2.shape
# Create ORB detector with 5000 features. 
orb_detector = cv2.ORB_create(nfeatures = 5000)
# Find keypoints and descriptors.  
# The first arg is the image, second arg is the mask 
#  (which is not reqiured in this case). 
kp1, d1 = orb_detector.detectAndCompute(img1, None) 
kp2, d2 = orb_detector.detectAndCompute(img2, None) 

test1 = cv2.drawKeypoints(img1, kp1, outImage =None, color=(0,255,0), flags=0)
test2 = cv2.drawKeypoints(img2, kp2, outImage =None, color=(0,255,0), flags=0)


fig = plt.figure(figsize=(1,2))
fig.add_subplot(1,2,1)
plt.imshow(test1)
fig.add_subplot(1,2,2)
plt.imshow(test2)
plt.show()
"""  
# Match features between the two images. 
# We create a Brute Force matcher with  
# Hamming distance as measurement mode. 
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True) 
  
# Match the two sets of descriptors. 
matches = matcher.match(d1, d2) 
  
# Sort matches on the basis of their Hamming distance. 
matches.sort(key = lambda x: x.distance) 
  
# Take the top 90 % matches forward. 
matches = matches[:int(len(matches)*90)] 
no_of_matches = len(matches) 
  
# Define empty matrices of shape no_of_matches * 2. 
p1 = np.zeros((no_of_matches, 2)) 
p2 = np.zeros((no_of_matches, 2)) 
  
for i in range(len(matches)): 
  p1[i, :] = kp1[matches[i].queryIdx].pt 
  p2[i, :] = kp2[matches[i].trainIdx].pt 
  
# Find the homography matrix. 
homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC) 
  
# Use this matrix to transform the 
# colored image wrt the reference image. 
transformed_img = cv2.warpPerspective(img1_color, 
                    homography, (width, height)) 
  
# Save the output. 
cv2.imwrite('output.png', transformed_img) 
"""