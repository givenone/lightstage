from pystackreg import StackReg
from skimage import io
import numpy as np
import os

input_path = "./"
input_names = ["x", "x_c", "y", "y_c", "z", "z_c"]
input_format = ".bmp"

ref_path = "./"
ref_name = "w"
ref_format = ".bmp"

output_path = input_path + 'xformed/'
if not os.path.exists(output_path):
    os.mkdir(output_path)

ref_name = ref_path + ref_name + ref_format
ref = io.imread(ref_name)

for name in input_names:
    print(name)
    #load an input image
    input_file = input_path + name + input_format
    input_img = io.imread(input_file)

    #Bilinear transformation
    sr = StackReg(StackReg.BILINEAR)
    #sr = StackReg(StackReg.AFFINE)
    #sr = StackReg(StackReg.TRANSLATION)
    #sr = StackReg(StackReg.RIGID_BODY)
    #sr = StackReg(StackReg.SCALED_ROTATION)
    #xformed = sr.register_transform(ref, input_img)  <-- accepts only 1 channel two dimensional inputs

    xformed = np.zeros_like(input_img)
    sr.register(ref[:, :, 2], input_img[:,:, 2])
    xformed[:,:, 0] = sr.transform(input_img[:,:, 0])
    xformed[:,:, 1] = sr.transform(input_img[:,:, 1])
    xformed[:,:, 2] = sr.transform(input_img[:,:, 2])

    out_file = output_path + name + input_format
    io.imsave(out_file, xformed)

print("Alignment finished.")