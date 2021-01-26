import gc
import os
from time import sleep
from run_IS.network import U_Net
import cv2
import numpy as np
import torch
import tqdm
from skimage import io, img_as_ubyte, img_as_float32
from split_image import split, join_img
from unet_tools import *


#  vessels
mod = find_newest(r"C:\Users\USER\Documents\studia\zaklad\EC_rainbow\run_IS\models\vessels")
mod = torch.load(mod)

unet = U_Net(img_ch=1, output_ch=1)
unet.load_state_dict(mod)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
unet.to(device)

transf = [T.ToTensor()]
transf = T.Compose(transf)



r = r"D:\EC_rainbow_data\10092020RAINBOW EXP ii"
file = r"Rainbow_d21_AE0088_2_final_2020_09_07__09_41_03 zeiss_Mip"
test = io.imread(os.path.join(r, file, file+".tif"), plugin="tifffile")
test = img_as_ubyte(test[:, :, 3])
smooth_umap_predict_vessels(test, unet, os.path.join(r, file, "vessels_mask.tif"), transf)













# Epoch [97/100], Loss: 1.9275,
# [Training] Acc: 0.4705, SE: 0.2336, SP: 0.4736, PC: 0.3952, F1: 0.2788, JS: 0.2064, DC: 0.2788
# Decay learning rate to lr: 5.262394954241189e-05.
# [Validation] Acc: 0.4902, SE: 0.3009, SP: 0.4927, PC: 0.3453, F1: 0.2987, JS: 0.2228, DC: 0.2987
# Best U_Net model score : 0.5215


# cells
mod = find_newest(r"C:\Users\USER\Documents\studia\zaklad\EC_rainbow\run_IS\models\cells")
mod = torch.load(mod)

unet = U_Net(img_ch=4, output_ch=3)
unet.load_state_dict(mod)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
unet.to(device)

transf = [T.ToTensor()]
transf = T.Compose(transf)


r = r"C:\Users\USER\Documents\studia\zaklad\EC_rainbow\cells"

files = os.listdir(r)
files = [file for file in files if file.startswith("test_image")]

# file = "test_image3.tif"
data = [io.imread(os.path.join(r, file), plugin="tifffile") for file in files]
data = [img_as_ubyte(img) for img in data]

pred = [pred_f_way(img, unet, 3, transf, rot_f_pred) for img in data]


for i, img in enumerate(pred):
    io.imsave(fname=os.path.join(r, "prediction" + str(i+1) + ".tif"), arr=img, plugin="tifffile", imagej=True)




r = r"D:\EC_rainbow_data\10092020RAINBOW EXP ii"
file = r"Rainbow_d21_AE0088_2_final_2020_09_07__09_41_03 zeiss_Mip"
dat = io.imread(os.path.join(r, file, file+".tif"), plugin="tifffile")
dat = img_as_ubyte(dat)
dat = pred_f_way(dat, unet, 3, transf, rot_f_pred)
io.imsave(fname=os.path.join(r, file, "cells_pred.tif"), arr=dat, plugin="tifffile", imagej=True)







# edges
r = r"C:\Users\USER\Documents\studia\zaklad\EC_rainbow\run_IS\models\edges"
files = (os.listdir(r))
files = [os.path.join(r, x) for x in files]
mod = max(files, key=os.path.getctime)
mod = torch.load(mod)


unet = U_Net(img_ch=4, output_ch=3)
unet.load_state_dict(mod)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
unet.to(device)

transf = [T.ToTensor()]
transf = T.Compose(transf)

targ = r"C:\Users\USER\Documents\studia\zaklad\EC_rainbow\cell_edges"
r = r"C:\Users\USER\Documents\studia\zaklad\EC_rainbow\cells"
files = os.listdir(r)
files = [file for file in files if file.startswith("test_image")]
data = [io.imread(os.path.join(r, file), plugin="tifffile") for file in files]
data = [img_as_ubyte(img) for img in data]

pred = [pred_f_way(img, unet, 3, transf, rot_f_pred) for img in data]

for i, img in enumerate(pred):
    io.imsave(fname=os.path.join(targ, "prediction" + str(i+1) + ".tif"), arr=img, plugin="tifffile", imagej=True)



