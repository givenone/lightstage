import cv2 as cv
import numpy as np
from numpy import array
from PIL import Image
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import math
import sys,os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from scipy.spatial.transform import Rotation as Rot
import concurrent.futures

"""
plot resulting image using matplotlib.pyplot
Get input as np array.
It copies original input to prevent it from being changed.
Then it normalizes input to make xyz normal to rgb image.
< normal map >
The resulting x,y,z normal map is converted to r,g,b image.
"""

def plot(image) :
    #Input : np array
    a = image.shape
    height, width = a[0], a[1]
    copied_image = np.copy(image)
    if(len(a) == 3) :
        for h in range(height):
            normalize(copied_image[h], copy=False)  #normalizing

        copied_image = (image + 1.0) / 2.0
        copied_image *= 255.0
        im = Image.fromarray(copied_image.astype('uint8'))

        plt.imshow(im)
        plt.show()

"""
save resulting image as a file
"""
def save(path, form, image) :

    a = image.shape
    if(len(a) == 3) :
        copied_image = np.copy(image)
        copied_image = (copied_image + 1.0) / 2.0
        copied_image *= 255.0
        im = Image.fromarray(copied_image.astype('uint8'))
        im.save(path+form)
    else :
        im.save(path+form)

"""
save resulting normal map as binary
"""
def save_bin(path, form, image) :

    height, width, _ = image.shape
    header = '''normal
    format binary_little_endian 1.0
    num_cols %d
    num_rows %d
    property float nx
    property float ny
    property float nz
    end_header'''%(width, height)

    file = open(path+form,"wb")
    file.write(header.encode('ascii'))
    file.write(b'\n')
    
    for h in range(height) :
        for w in range(width) :
            file.write(np.dtype('<f4').type(image[h][w][0]))
            file.write(np.dtype('<f4').type(image[h][w][1]))
            file.write(np.dtype('<f4').type(image[h][w][2]))

    print("write ply done")
    file.close()


"""
mixed albedo calculation.
Mixed(Diffuse + specular) albedo is obtained by adding binary spherical gradients and their complement.
"""    
def calculateMixedAlbedo(images) :
    
    sum_img = np.zeros_like(images[0])
    for i in images :
        sum_img += i


    print("Mixed Albedo Done")

    return sum_img/3 # BGR Image


"""
simply subtract the estimated specular reflectance ρ 
from the mixed albedo (obtained from the sum of a gradient and its complement) 
to obtain our separated diffuse albedo
"""
def calculateDiffuseAlbedo(mixed, specular) :
    
    out_img = np.zeros_like(mixed).astype('float32')
    out_img[...,0] = np.subtract(mixed[...,0], specular)
    out_img[...,1] = np.subtract(mixed[...,1], specular)
    out_img[...,2] = np.subtract(mixed[...,2], specular)
    out_img /= 2
    out_img = np.clip(out_img, 0, 255)
    #median = cv.medianBlur(out_img, 5) # each channel independently.
    out_img = cv.GaussianBlur(out_img, (21, 21), cv.BORDER_DEFAULT)
    #blur = cv.bilateralFilter(out_img,9,75,75)
    print("Diffuse Albedo Done")

    return out_img# BGR


""" 
color-space separation for the albedo under uniform illumination.
convert
observed RGB colors of a surface under a binary spherical gradient and its complement 
(R g , G g , B g and R c , G c , B c respectively) into HSV space 
(H g , S g ,V g and H c , S c ,V c respectively). 
Assuming the gradient is brighter than its complement (V g > V c ), 
we formulate the amount of specular reflectance mixed in 
the bighter gradient to be equal to the amount δ that needs to be subtracted from 0
R g , G g , B g such that its new saturation S g 
equals the saturation of the complementary gradient S c

We choosed arbitrary coefficient "128" to use δ as an albedo.
"""

def calculateSpecularAlbedo(images, imgs) :

    H, S, V  = [], [], []

    for img in imgs:
        # H S V Separation
        hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV) #HSV
        h, s, v = cv.split(hsv_img)        
        arr = array(h)
        H.append(arr.astype('float32'))
        arr = array(s)
        S.append(arr.astype('float32'))
        arr = array(v)
        V.append(arr.astype('float32'))


    height, width, _ = images[0].shape
    
    specular_albedo = [None,None,None] #original

    for i in range(3) :
        # hsv 
        h_g, s_g, v_g, h_c, s_c, v_c = H[2*i], S[2*i], V[2*i], H[2*i+1], S[2*i+1], V[2*i+1]        
        
        # chroma
        b,g,r = cv.split(images[2*i])
        b_c, g_c,r_c = cv.split(images[2*i+1])
        c_g =  np.subtract(np.maximum(np.maximum(r, g), b), np.minimum(np.minimum(r, g), b))
        c_c =  np.subtract(np.maximum(np.maximum(r_c, g_c), b_c), np.minimum(np.minimum(r_c, g_c), b_c)) 

        t = np.divide(c_g, s_c, out=np.zeros_like(c_g), where=s_c!=0)
        spec = np.subtract(v_g, t*128) 
        spec = np.clip(spec, 0, 255)

        t = np.divide(c_c, s_g, out=np.zeros_like(c_c), where=s_g!=0)
        spec_c = np.subtract(v_c, t*128) 
        spec_c = np.clip(spec_c, 0, 255)

        mask = v_g > v_c
        # need to mask out error (when saturation is bigger in bright side)
        specular_albedo[i] = np.empty_like(v_g)
        specular_albedo[i][mask] = spec[mask]
        specular_albedo[i][~mask] = spec_c[~mask]

    
    # using normailzed max values as the specular
    specular_max = np.median(specular_albedo, axis=0)
    min_val = np.min(specular_max)
    max_val = np.max(specular_max)

    specular_max = (specular_max - min_val) / (max_val - min_val) * 255.0 # 255 is too bright
    specular_max = np.clip(specular_max, 0, 255)
    print("Specular Albedo Done")  
    
    plt.title("specular_albedo")
    #median = cv.medianBlur(specular_max, 5)

    specular_max = cv.GaussianBlur(specular_max, (21, 21), cv.BORDER_DEFAULT)
    #median = cv.bilateralFilter(specular_max,9,75,75) # each channel independently.
    return specular_max

"""
The binary spherical gradients and their complements can be directly 
employed to compute photometric surface normals
"""
def calculateMixedNormals(images):
 
    height, width, _ = images[0].shape

    N_x = (images[0] - images[1])
    N_y = (images[2] - images[3])
    N_z = (images[4] - images[5])

    encodedImage = np.empty_like(N_x).astype('float32')
    encodedImage[...,0] = N_x[..., 0] #Mixed Normal -> blue component
    encodedImage[...,1] = N_y[..., 0]
    encodedImage[...,2] = N_z[..., 0]

    for h in range(height):
            normalize(encodedImage[h], copy=False)  #normalizing

    print("Mixed Normal Done")
    return encodedImage


"""
transform the acquired binary spherical gradients and their complements 
from RGB to Mallick’s proposed suv space. 
This transformation rotates the RGB color information to align 
any white component in the original RGB signal with the s component,
while leaving the chroma in the u, v components. 
The chroma in colored dielectric materials is the result of
absorption due to subsurface scattering and hence the u, v components 
can now be employed to recompute the photometric normals
"""

def get_D_quat(unit_vec = None) : # get magnitude of diffuse component in rgb space 

    if unit_vec == None :
        v1 = [1.0, 0.0, 0.0] # any axis
    else:
        v1 = unit_vec
    v2 = [1.0, 1.0, 1.0]    # white vector
    v2 = v2 / np.linalg.norm(v2)

    rot_axis = np.cross(v1, v2) # a x b
    rot_angle = np.arccos(np.dot(v1, v2))
    cos_val = np.cos(rot_angle/2.0)
    sin_val = np.sin(rot_angle/2.0)
 
    rot1 = Rot.from_quat([rot_axis[0]*sin_val, rot_axis[1]*sin_val, rot_axis[2]*sin_val, cos_val])    
  
    return Rot.inv(rot1).as_matrix()


"""
calculate diffuse normal using get_D_quat function
"""

def calculateDiffuseNormals(images):

    height, width, _ = images[0].shape

    rot_vec = [1,1,1] # specular : white component
    I = get_D_quat() # rotation matrix    
    N = []

    for i in range(3) :
        I_suv_g = np.dot(images[2*i], I.T) 
        I_suv_c = np.dot(images[2*i+1], I.T)
        G=np.sqrt(I_suv_g[:,:,1]**2 + I_suv_g[:,:,2]**2)
        G_C=np.sqrt(I_suv_c[:,:,1]**2 + I_suv_c[:,:,2]**2)
        N.append(G-G_C)

    encodedImage = np.empty_like(images[0]).astype('float32')
    encodedImage[..., 0] = N[0] # X
    encodedImage[..., 1] = N[1] # Y
    encodedImage[..., 2] = N[2] # Z

    for h in range(height):
        normalize(encodedImage[h], copy=False)  #normalizing
    
    print("Diffuse Normal Done")  

    return encodedImage

"""
mixed normals encode a mixture of a diffuse normal and some specular reflectance.
The specular reflectance can be computed by decomposing weighted sum of mixed normal and diffuse normal.
The specular normal is then calculated as the sum of the viewing direction and the estimated reflection vector
"""

def calculateSpecularNormals(diffuse_albedo, specular_albedo, mixed_normal, diffuse_normal, viewing_direction) : 
    
    su = specular_albedo + diffuse_albedo[...,0]
    alpha = np.divide(diffuse_albedo[...,0], su, out=np.zeros_like(su), where=su!=0)
    d_x = np.multiply(diffuse_normal[..., 0], alpha)
    d_y = np.multiply(diffuse_normal[..., 1], alpha)
    d_z = np.multiply(diffuse_normal[..., 2], alpha)
    alphadiffuse = np.empty_like(diffuse_normal).astype('float32')
    alphadiffuse[..., 0] = d_x
    alphadiffuse[..., 1] = d_y
    alphadiffuse[..., 2] = d_z

    # testing...
    Reflection = np.subtract(mixed_normal, alphadiffuse)
    #np.mean(specular_albedo
    #Reflection = np.subtract(mixed_normal, 0.1 * diffuse_normal)
    # testing done ...
    height, width, _ = Reflection.shape
    for h in range(height):
        normalize(Reflection[h], copy=False) # Normalize Reflection

    normal = np.add(Reflection, -viewing_direction).astype('float32')
    for h in range(height):
        normalize(normal[h], copy=False)

    print("Speuclar Normal Done")

    return normal

from scipy.ndimage import gaussian_filter
def HPF(normal) : # High Pass Filtering for specular normal reconstruction
    
    height, width, _ = normal.shape
    
    #blur = gaussian_filter(normal, sigma = 101)

    blur = np.empty_like(normal)
    blur[..., 0] = gaussian_filter(normal[..., 0], sigma=101)
    blur[..., 1] = gaussian_filter(normal[..., 1], sigma=101)
    blur[..., 2] = gaussian_filter(normal[..., 2], sigma=101)
    filtered_normal = cv.subtract(normal, blur)

    for h in range(filtered_normal.shape[0]) :
        normalize(filtered_normal[h], copy = False)

    print("High Pass Filter Done")

    return filtered_normal


def synthesize(diffuse_normal, filtered_normal) :

    syn = np.add(diffuse_normal, 0.3 * filtered_normal)

    height, width, _ = syn.shape

    for h in range(height):
        normalize(syn[h], copy=False)
    
    print("Specular Normal Synthesis Done")

    return syn

def generate_viewing_direction(shape, focalLength, sensor = (0.036, 0.024)) :
    
    height, width, _ = shape
    
    centerX = width/2
    centerY = height/2
    sensor_width = sensor[0]
    sensor_height = sensor[1]
    x_pitch = sensor_width/width
    y_pitch = sensor_height/height

    vd = [ [ ( (float)(x-centerX) * x_pitch,
            (float)(centerY-y) * y_pitch,
            -focalLength)
            for x in range(width)]
            for y in range(height)]

    v = np.array(vd)
    vd = v.astype('float32')    
    # Normalization
    for h in range(height) :
        normalize(vd[h], copy = False)

    # ROTATION 
    #r = Rot.from_euler('y', math.atan(3/3.9)) ###### TODO :: atan(x/z) (the coordinate of camera.)
    #vd = r.apply(vd.reshape(-1,3)).reshape((height,width,3))

    print("Viewing Direction Done")
    return vd

def execute(item) :
    
    directory, path, form = item
    path = path + directory + "/"
    names = ["x", "x_c", "y", "y_c", "z", "z_c"]
    names = [path + name + form for name in names]
    
    images = []
    imgs = []
    for name in names :
        img = cv.imread(name, 3) #BGR
        img = cv.transpose(img) # transpose to change width and height. Captured image is rotated.
        img = cv.flip(img, 1)
        """
        TODO ::

        align input images with scanner imagees

        """
        imgs.append(img)
        arr = array(img).astype('float32')
        images.append(arr)

    mixed_normal = calculateMixedNormals(images)
    diffuse_normal = calculateDiffuseNormals(images)
    filtered_normal = HPF(mixed_normal)
    syn = synthesize(diffuse_normal, filtered_normal)
    
    view_flag = False

    if view_flag :
        plt.title("diffuse normal")
        plot(diffuse_normal)

        plt.title("Synthesized")
        plot(syn)

    save_flag = True

    if save_flag :
        
        dirname = os.path.join(os.path.abspath(os.path.dirname(__file__)) + "/" + path, 'result')
        #print(dirname)
        if not os.path.exists(dirname):
            #print(dirname)
            os.makedirs(dirname)
        #testing...

        from tifffile import imsave
        print(syn[2000][1000], path)
        syn = cv.cvtColor(syn, cv.COLOR_BGR2RGB)
        imsave(path+'result/normal.tif', syn)
        #save_bin(path+"result/syn_mixed", ".png", syn)
        #np.save('test.npy', syn)

    # python binary_reconstruction.py [-V] [-format png] [-path ./input_image]


if __name__ == "__main__":
    import argparse, configparser
    parser = argparse.ArgumentParser(description='''
    This script reads a images caputured with light stage and binary spherical gradient pattern and
    generate diffuse / specular albedo and normal map.
    The resulting normal map and diffuse albedo is saved. 
    ''')
    parser.add_argument('-V', '-v', 
    dest = 'visualizing', 
    action = 'store_true', 
    required=False,
    help='visualising reconstruction result')

    parser.add_argument('-format',  
    dest = 'form',  
    required=True,
    help='input image format')

    parser.add_argument('-path',  
    dest = 'path',  
    required=True,
    help='input image path')
    args = parser.parse_args()
    
    path = args.path
    form = "." + args.form

    with concurrent.futures.ProcessPoolExecutor() as executor:
            executor.map(execute, [("Front", path, form), ("Left", path, form), ("Right", path, form)])
    print("normal reconstruction done")
    
 