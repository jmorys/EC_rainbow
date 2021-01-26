import argparse
import os
from run_IS.network import U_Net
import torch
from skimage import io, img_as_ubyte, img_as_float32
from split_image import split, join_img
from unet_tools import *
import tqdm


def main(config):

    if os.path.isfile(config.model_location):
        mod = config.model_location
    else:
        mod = find_newest(r"C:\Users\USER\Documents\studia\zaklad\EC_rainbow\run_IS\models\vessels")
    mod = torch.load(mod)
    images = []
    names = []
    if os.path.isfile(config.input):
        images.append(config.input)
        names.append(config.input.split("\\")[-1][:-4])
    else:
        files = [file for file in os.listdir(config.input) if file.endswith(".tif")]
        images.extend([os.path.join(config.input, file) for file in files])
        names.extend([file[:-4] for file in files])

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")


    if config.segment == "vessels":
        unet = U_Net(img_ch=1, output_ch=1)
        unet.load_state_dict(mod)
        unet.to(device)
        transf = [T.ToTensor()]
        transf = T.Compose(transf)

        for image_path, name in tqdm.tqdm(zip(images, names)):
            image = io.imread(image_path, plugin="tifffile")
            image = img_as_ubyte(image[:, :, 3])
            out_location = os.path.join(config.out_location, name + "_vessels.tif")
            smooth_umap_predict_vessels(image, unet, out_location, transf)

    elif config.segment == "cells":
        unet = U_Net(img_ch=1, output_ch=1)
        unet.load_state_dict(mod)
        unet.to(device)
        transf = [T.ToTensor()]
        transf = T.Compose(transf)
        for image_path, name in tqdm.tqdm(zip(images, names)):
            image = io.imread(image_path, plugin="tifffile")
            image = img_as_ubyte(image)
            image = pred_f_way(image, unet, 3, transf, rot_f_pred)
            out_location = os.path.join(config.out_location, name + "_vessels.tif")
            smooth_umap_predict_vessels(image, unet, out_location, transf)
            io.imsave(fname=out_location, arr=image, plugin="tifffile", imagej=True)
    else:
        print("wrong segmentation type")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_location', type=str)
    parser.add_argument('--input', type=str)
    parser.add_argument('--out_location', type=str)
    parser.add_argument('--segment', type=str, default="vessels", help="vessels/cells")
    config = parser.parse_args()
    main(config)
