import cv2
import dlib
import matplotlib.pyplot as plt
# set up the 68 point facial landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# bring in the input image
img = cv2.imread('LSMK_0603_103015\\Front\\z.bmp', 1)
img = cv2.transpose(img) # transpose to change width and height. Captured image is rotated.
img = cv2.flip(img, 1)
# convert to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# detect faces in the image
faces_in_image = detector(img_gray, 0)


# bring in the input image
img2 = cv2.imread("LSMK_0603_103015\\Front\\x.bmp", 1)
img2 = cv2.transpose(img2) # transpose to change width and height. Captured image is rotated.
img2 = cv2.flip(img2, 1)
# convert to grayscale
img_gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# detect faces in the image
faces_in_image_2 = detector(img_gray2, 0)


landmarks_list_1 = []
landmarks_list_2 = []

# loop through each face in image
for face in faces_in_image:
	print('1')
	# assign the facial landmarks
	landmarks = predictor(img_gray, face)
	
	# unpack the 68 landmark coordinates from the dlib object into a list 
	
	for i in range(0, landmarks.num_parts):
		landmarks_list_1.append((landmarks.part(i).x, landmarks.part(i).y))

	# for each landmark, plot and write number
	for landmark_num, xy in enumerate(landmarks_list_1, start = 1):
		cv2.circle(img, (xy[0], xy[1]), 12, (168, 0, 20), -1)
		cv2.putText(img, str(landmark_num),(xy[0]-7,xy[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(255,255,255), 1)

for face in faces_in_image_2:
	print('2')
	# assign the facial landmarks
	landmarks = predictor(img_gray2, face)
	
	# unpack the 68 landmark coordinates from the dlib object into a list 
	for i in range(0, landmarks.num_parts):
		landmarks_list_2.append((landmarks.part(i).x, landmarks.part(i).y))

	# for each landmark, plot and write number
	for landmark_num, xy in enumerate(landmarks_list_2, start = 1):
		cv2.circle(img2, (xy[0], xy[1]), 12, (168, 0, 20), -1)
		cv2.putText(img2, str(landmark_num),(xy[0]-7,xy[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(255,255,255), 1)


orb_detector = cv2.ORB_create()
kp1=[]
kp2=[]

for x, y in landmarks_list_1 :
	kp1.append(cv2.KeyPoint(x, y, 2))
for x, y in landmarks_list_2 :
	kp2.append(cv2.KeyPoint(x, y, 2))

kp1, d1 = orb_detector.compute(img, kp1)
kp2, d2 = orb_detector.compute(img2, kp2)
test1 = cv2.drawKeypoints(img, kp1, outImage =None, color=(0,255,0), flags=0)
test2 = cv2.drawKeypoints(img2, kp2, outImage =None, color=(0,255,0), flags=0)
#print(d1, d2)

# visualise the image with landmarks
#cv2.namedWindow('img',cv2.WINDOW_NORMAL)
#cv2.resizeWindow('img', 800,600)
#cv2.imshow('img',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#rint(landmarks_list_1, landmarks_list_2)

fig = plt.figure(figsize=(1,2))
fig.add_subplot(1,2,1)
plt.imshow(test1)
fig.add_subplot(1,2,2)
plt.imshow(test2)
plt.show()



bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(d1,d2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 10 matches.
img3 = cv2.drawMatches(img,kp1,img2,kp2,matches[:5], flags=2, outImg =None)

plt.imshow(img3),plt.show()
