from stl import mesh
import math
import numpy as np
import sys
import os
import argparse
import PIL.Image as pil
import matplotlib.pyplot as plt

import test_simple_copy

def parse_args():
    parser = argparse.ArgumentParser(
        prog = '3D Model generator from depth map',
        description = 'Attempts to generate a 3D model from input depth map')
    
    parser.add_argument('--image_path', type = str,
                        help = 'path to a test depth map or folder of depth maps')
    
    parser.add_argument('--ext', type = str,
                        help = 'extension to search in folder', default = "jpg")
    
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')
    
    parser.add_argument("--pred_metric_depth",
                        help='if set, predicts metric depth instead of disparity. (This only '
                             'makes sense for stereo-trained KITTI models).',
                        action='store_true')
    
    parser.add_argument('--model_name', type=str,
                        help = 'name of a pretrained model to use',
                        choices = [
                            "mono_640x192",
                            "stereo_640x192",
                            "mono+stereo_640x192",
                            "mono_no_pt_640x192",
                            "stereo_no_pt_640x192",
                            "mono+stereo_no_pt_640x192",
                            "mono_1024x320",
                            "stereo_1024x320",
                            "mono+stereo_1024x320"],
                        default = 'mono_640x192')
    
    return parser.parse_args()

def modelgen(args):
    """Function to generate a 3D model for a single image or folder of images
    """
    print('Default model is used (mono_640x192)')
    
    #depth_map = test_simple_copy.test_simple(args)
    print(depth_map.shape)
    
    print('Yeehoooo')
    
def value_printer(img_path):
    
    # Load the image using Pillow
    img = pil.open(img_path)

    # Convert the image to a NumPy array
    img_array = np.array(img)
    print(img_array[234][637])
    print(img_array.shape)

def pixel_depth_interpretor(pixel):
    # We want to print within 21cm*29,7cm*5cm at 150 ppi
    # That makes around 1246*1754*296
    # Maximum depth (point is considere the furthest) when pixel value is 0, setting it to 0
    relative_depth = np.mean(pixel)
    scaled_pixel = relative_depth/(1+math.exp(5-0.08*relative_depth))
    #depth_in_obj = int(relative_depth*296/255)
    #return depth_in_obj
    #return relative_depth
    return scaled_pixel

def depth_interpreted_image(img_path):
    # Load the image using Pillow
    img = pil.open(img_path)

    # Convert the image to a NumPy array
    img = np.array(img)
    
    new_img = np.zeros((img.shape[0], img.shape[1]))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            new_img[i][j] = pixel_depth_interpretor(img[i][j])
    
    return new_img
    
def img_convertor(img_path, goal_res):
    
    img = depth_interpreted_image(img_path)
    
    # Converts the image to desired resolution
    # Goal res is 1246*1754*296
    (x, y) = img.shape[0], img.shape[1]
    # if x > goal_res[0]: # Input image resolution is bigger than desired resolution
    #     if (x/goal_res[0]) > (y/goal_res[1]):
    #         rel = x/goal_res[0]
    #         new_img = np.zeros((goal_res[0], math.ceil(y/rel))) 
    #     else:
    #         rel = y/goal_res[1]
    #         new_img = np.zeros((math.ceil(x/rel), goal_res[1]))
    #     new_img = img[::math.floor(rel)][::math.floor(rel)]
    # if x < goal_res[0] : # Input image resolution is smaller than desired resolution
    #     if (x/goal_res[0]) > (y/goal_res[1]):
    #         rel = goal_res[0]/x
    #         #new_img = np.zeros((goal_res[0], math.ceil(y*rel))) 
    #         new_img = np.zeros((x,y))
    #     else:
    #         rel = goal_res[1]/y
    #         #new_img = np.zeros((math.ceil(x*rel), goal_res[1]))
    #         new_img = np.zeros((x,y))
    #     for i in range(x):
    #         for j in range(y):
    #             #new_img[math.floor(i*rel)][math.floor(j*rel)] = img[i][j]
    #             new_img[i][j] = img[i][j]
    
    # im = pil.fromarray(new_img.astype(np.uint8))
    im = pil.fromarray(img.astype(np.uint8))
    name_dest_im = os.path.join("/home/vinhsp/SRD-Depth/test", "{}_disp3.jpeg".format("test"))
    im.save(name_dest_im)
    # print(new_img.shape)
    # return new_img
    return 

# def mesh_gen(depth_map):
    
    
# def mesh(img):
#     vertices = np.zeros((img.shape[0], img.shape[1]))
#     for row in img:
#         for column in row:
#             vertices[row][column] = [0, 0, 0]
#     return
    
if __name__ == '__main__':
    #args = parse_args()
    #modelgen(args)
    if len(sys.argv) > 1:
        img = sys.argv[1]
        #value_printer(img)
        all_depth_values(img)
        img_convertor(img, (1246, 1754))
    else:
        print("Please provide a name as a command line argument.")