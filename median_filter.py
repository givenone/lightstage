import cv2 as cv
import numpy as np
from scipy import ndimage, misc
import matplotlib.pyplot as plt
import sys,os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

path = "C:\\Users\\yeap98\\Desktop\\lightstage\\LSMK_0508_170205\\"
form = ".bmp"

names = ["x", "x_c", "y", "y_c", "z", "z_c", "w"]

images = []
dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = os.path.join(dir_path, "filtered_image")

for na in names :
    name = path + na + form
    img = cv.imread(name, -1) #BGR
    fig = plt.figure()
    #ax1 = fig.add_subplot(121)  # left side
    #ax2 = fig.add_subplot(122)
    median = cv.medianBlur(img, 11) # each channel independently.

    #rgb = cv.cvtColor(median, cv.COLOR_BGR2RGB)
    pa = dir_path + '/{}.bmp'.format(na)
    print(pa)
    cv.imwrite(pa, median)
    #ax1.imshow(img)
    #ax2.imshow(median)
    #plt.show()
    #images.append(img)