import math
import numpy as np
import sys
import os
import argparse
import PIL.Image as pil
import glob
import matplotlib.pyplot as plt

from models import utils
import torch
from torchvision import transforms, datasets
from stl import mesh

def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for Monodepthv2 models.')

    parser.add_argument('--image_path', type=str,
                        help='path to a test image or folder of images', required=True)
    parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder', default="jpg")
    parser.add_argument('--model_name', type=str,
                        help='name of a pretrained model to use')
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')
    return parser.parse_args()

def load_image(args):
    img = pil.open(args.image_path).convert('RGB')
    og_width, og_height = img.size
    return img

def load_model(args):
    utils.download_model_if_doesnt_exist(args.model_name)
    model_path = os.path.join("models", args.model_name)
    print("-> Loading model from ", model_path)
    model = None
    return model

def depth_est(paths, output_directory, model, args):
    with torch.no_grad():
        for idx, image_path in enumerate(paths):
            if image_path.endswith("_disp.jpg"):
                continue
            
            input_image = load_image(image_path)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)
            
            input_image = input_image.to(device)
    return

def set_dir_path(args):
    if os.path.isfile(args.image_path):
        paths = [args.image_path]
        output_directory = os.path.dirname(args.image_path)
    
    elif os.path.isdir(args.image_path):
        paths = glob.glob(os.path.join(args.image_path, '*.jpg'))
        output_directory = args.image_path
    else:
        raise Exception("Can't find image or path : {}".format(args.image_path))
    return paths, output_directory



def magma_to_depth(magma_map):
    magma_map = pil.open(magma_map)
    magma_map = np.array(magma_map)
    magma_map = magma_map / 255.0
    cmap = plt.get_cmap('magma')
    
    img = magma_map.reshape(-1, magma_map.shape[-1])
    unique_colors = np.unique(img, axis=0)
    
    sampled_colors = cmap(np.linspace(0, 1, unique_colors.shape[0]))[:, :3]
    
    normalized_depth = np.zeros(magma_map.shape[:2])
    print(normalized_depth.shape)
    for i in range(magma_map.shape[0]):
        for j in range(magma_map.shape[1]):
            pixel_color = magma_map[i, j]
            differences = np.linalg.norm(sampled_colors - pixel_color, axis=1)
            closest_color_index = np.argmin(differences)
            normalized_depth[i, j] = closest_color_index / (unique_colors.shape[0] - 1)
            normalized_depth[i, j] = normalized_depth[i, j]/(1 + math.exp(.5-5*normalized_depth[i, j]))
    return normalized_depth
    
def generate_3d(int_depth_map):
    # input_img = pil.open(depth_map)
    img = np.array(int_depth_map)
    (x, y) = img.shape
    
    vertices = np.zeros((x*y, 3))
    for i in range(x):
        for j in range(y):
            vertices[(i*y+j)] = [i, j, img[i][j]]
    
    faces = []
    for i in range(x-1):
        for j in range(y-1):
            faces.append([j + i*y, j+1 + i*y, j+(i+1)*y])
            faces.append([j+1 + i*y, j + (i+1)*y, j+1 +(i+1)*y])
    faces = np.array(faces)
    
    mesh_3d = mesh.Mesh(np.zeros(faces.shape[0], dtype = mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            mesh_3d.vectors[i][j] = vertices[f[j], :]
    return mesh_3d

def save_3d(mesh, path, output_dir):
    output_name = os.path.splitext(os.path.basename(path))[0]
    save_dest = os.path.join(output_dir, "{}.stl".format(output_name))
    mesh.save(save_dest)
    print("-> 3D model generated!")
    
def pipeline(args):
    # img = load_image(args.image_path)
    
    # model = load_model()
    # depth_map = depth_est(model, img)
    paths, output_dir = set_dir_path(args)
    print(paths, output_dir)
    
    for img in paths:  
        int_depth_map = magma_to_depth("/home/vinhsp/SRD-Depth/assets/test_image_disp.jpeg") 
        
        mesh = generate_3d(int_depth_map)
        save_3d(mesh, img, output_dir)
        
    return

if __name__ == '__main__':
    args = parse_args()
    pipeline(args)
