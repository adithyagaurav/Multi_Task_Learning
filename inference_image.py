import matplotlib.pyplot as plt
from PIL import Image
import glob
import numpy as np
import cv2
import open3d as o3d
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.autograd import Variable
from Hydranet import Hydranet


hydranet = Hydranet(2,6)
ckpt = torch.load("ExpKITTI_joint.ckpt", map_location='cpu')
hydranet.enc.load_state_dict(ckpt["state_dict"], strict=False)
hydranet.dec.load_state_dict(ckpt["state_dict"], strict=False)
hydranet.eval()

IMG_SCALE  = 1./255
IMG_MEAN = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
IMG_STD = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))
CMAP = np.load('cmap_kitti.npy')
NUM_CLASSES = 6


def prepare_img(img):
    return (img * IMG_SCALE - IMG_MEAN) / IMG_STD


def pipeline(img):
    with torch.no_grad():
        img_var = Variable(torch.from_numpy(prepare_img(img).transpose(2,0,1)[None]), requires_grad=False).float()
        if torch.cuda.is_available():
            img_var = img_var.cuda()
        depth, segm = hydranet(img_var)
        segm = cv2.resize(segm[0].cpu().data.numpy().transpose(1,2,0), img.shape[:2][::-1], interpolation=cv2.INTER_CUBIC)
        depth = cv2.resize(depth[0].cpu().data.numpy().transpose(1,2,0), img.shape[:2][::-1], interpolation=cv2.INTER_CUBIC)
        segm = CMAP[segm.argmax(axis=2)].astype(np.uint8)
        depth = np.abs(depth).astype(np.uint8)
        return depth, segm

images_files = glob.glob('data/*.png')
idx = np.random.randint(0, len(images_files))

img_path = images_files[idx]
img = np.array(Image.open(img_path))
depth, segm = pipeline(img)

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30,20))
ax1.imshow(img)
ax1.set_title('Original', fontsize=30)
ax2.imshow(segm)
ax2.set_title('Predicted Segmentation', fontsize=30)
ax3.imshow(depth, cmap="plasma", vmin=0, vmax=40)
ax3.set_title("Predicted Depth", fontsize=30)
plt.show()

rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(segm), o3d.geometry.Image(depth), convert_rgb_to_intensity=False)
intrinsics = o3d.camera.PinholeCameraIntrinsic(width = 1242, height = 375, fx = 721., fy = 721., cx = 609., cy = 609.)
point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd,intrinsics)
o3d.io.write_point_cloud("test.pcd", point_cloud)

pcd = o3d.io.read_point_cloud("test.pcd")
pcd.transform([[1,0,0,0], [0,-1,0,0], [0,0,-1,0],[0,0,0,1]])
o3d.visualization.draw_geometries([pcd])