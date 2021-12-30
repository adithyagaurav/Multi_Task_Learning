import matplotlib.pyplot as plt
from PIL import Image
import glob
import numpy as np
import cv2
import matplotlib.cm as cm
import matplotlib.colors as co

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

def project_to_image(img, depth, seg):
    cmap_proj = plt.cm.get_cmap("hsv", 256)
    cmap_proj = np.array([cmap_proj(i) for i in range(256)])[:, :3] * 255
    img_copy = cv2.addWeighted(img, 0.8, seg, 1.5, 0.7)
    for i in range(depth.shape[0]):
        for j in range(depth.shape[1]):
            num = np.random.randint(1, 20)
            num1 = np.random.randint(1, 20)
            if i>100 and (j%num==0 or i%num1==0) and depth[i][j]>5:
                if num<8:
                    continue
                point_depth = depth[i][j]
                color = cmap_proj[int(510/point_depth),:]
                cv2.circle(img_copy, (j,i), 2, tuple(color), thickness=-1)
    return img_copy

def depth_to_rgb(depth):
    normalizer = co.Normalize(vmin=0, vmax=80)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='plasma')
    colormapped_im = (mapper.to_rgba(depth)[:, :, :3] * 255).astype(np.uint8)
    return colormapped_im

video_files = sorted(glob.glob("data/*.png"))
result_video = []
projected_video = []
count=0
for idx, img_path in enumerate(video_files):
    image = np.array(Image.open(img_path))
    h, w, _ = image.shape
    depth, seg = pipeline(image)
    projected_output = project_to_image(image, depth, seg)
    result_video.append(cv2.cvtColor(cv2.vconcat([image, seg, depth_to_rgb(depth)]), cv2.COLOR_BGR2RGB))
    projected_video.append(cv2.cvtColor(projected_output, cv2.COLOR_BGR2RGB))
    count+=1

out = cv2.VideoWriter('output/out.mp4',cv2.VideoWriter_fourcc(*'MP4V'), 15, (w,3*h))
proj = cv2.VideoWriter('output/proj.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 15, (w,h))

for i in range(len(result_video)):
    out.write(result_video[i])
out.release()


for i in range(len(projected_video)):
    proj.write(projected_video[i])
proj.release()