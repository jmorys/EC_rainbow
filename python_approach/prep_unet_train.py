import os
import sys
from skimage import io, img_as_ubyte, transform, morphology
import re
import random
import numpy as np
from split_image import split, join_img


def read_and_split(path, ltrain=1, lgt=1, rescale=None):
    image = io.imread(path)
    if rescale is not None:
        image = transform.rescale(image, (1, rescale, rescale), anti_aliasing=False)
    lab = image[ltrain:(ltrain+lgt), :, :]
    lab = lab >= 0.5
    lab = img_as_ubyte(lab)
    dat = img_as_ubyte(image[:ltrain, :, :])
    return dat, lab



def train_Unet(path, ltrain=1, lgt=1, rescale=None):
    os.chdir(path)
    files = os.listdir()
    files = [x for x in files if re.match("train_for_split.*.tif", x)]

    data_split = []
    label_split = []
    for file in files:
        data, label = read_and_split(file, ltrain, lgt, rescale)
        label = split(label, 256, 20)
        data = split(data, 256, 20)
        data_split.extend(data)
        label_split.extend(label)

    # random

    if len(data_split[0].shape) == 3:
        data_split = [np.swapaxes(np.swapaxes(x, 0, 1), 0, 1) for x in data_split]

    if len(label_split[0].shape) == 3:
        label_split = [np.swapaxes(np.swapaxes(x, 0, 2), 0, 1) for x in label_split]

    rand = random.sample(range(len(data_split)), len(data_split))
    test = 0.05
    test = int(len(rand)//(1/test))
    validate = 0.33
    validate = int(len(rand)//(1/validate))
    train = 0.72
    train = int(len(rand)//(1/train))

    validate = rand[:validate]
    train = rand[-train:]
    test = rand[:test]

    os.mkdir(".\\test\\")
    os.mkdir(".\\test_GT\\")
    os.mkdir(".\\train_GT\\")
    os.mkdir(".\\train\\")
    os.mkdir(".\\validate_GT\\")
    os.mkdir(".\\validate\\")

    for i in range(len(test)):
        l = test[i]
        io.imsave(fname=".\\test_GT\\tile"+str(i)+".tif", arr=label_split[l], plugin="tifffile", imagej=True)
        io.imsave(fname=".\\test\\tile" + str(i) + ".tif", arr=data_split[l], plugin="tifffile", imagej=True)

    for i in range(len(train)):
        l = train[i]
        io.imsave(fname=".\\train_GT\\tile"+str(i)+".tif", arr=label_split[l], plugin="tifffile", imagej=True)
        io.imsave(fname=".\\train\\tile" + str(i) + ".tif", arr=data_split[l], plugin="tifffile", imagej=True)

    for i in range(len(validate)):
        l = validate[i]
        io.imsave(fname=".\\validate_GT\\tile"+str(i)+".tif", arr=label_split[l], plugin="tifffile", imagej=True)
        io.imsave(fname=".\\validate\\tile" + str(i) + ".tif", arr=data_split[l], plugin="tifffile", imagej=True)


train_Unet(path=r"C:\Users\USER\Documents\studia\zaklad\EC_rainbow\vessels\unet", rescale=0.5)


train_Unet(path=r"C:\Users\USER\Documents\studia\zaklad\EC_rainbow\cells", ltrain=4, lgt=3)

r = r"C:\Users\USER\Documents\studia\zaklad\EC_rainbow\cells"
targ = r"C:\Users\USER\Documents\studia\zaklad\EC_rainbow\cell_edges"
files = [x for x in os.listdir(r) if (x.startswith("label") and x.endswith(".tif"))]
disc = morphology.disk(7)
disc_err = morphology.disk(5)
sources = [x for x in os.listdir(r) if (x.startswith("test_image") and x.endswith(".tif"))]
for i, (source_f, file) in enumerate(zip(sources, files)):
    image = io.imread(os.path.join(r, file), plugin="tifffile")
    image_dil = np.zeros(image.shape)
    for l, lay in enumerate(np.moveaxis(image, 2, 0)):
        dil = morphology.dilation(lay, disc)
        dil = morphology.erosion(dil, disc_err)
        dil = dil.astype(np.uint8) - lay
        image_dil[:, :, l] = morphology.dilation(dil)
    image_dil = image_dil.astype(np.uint8)
    source = io.imread(os.path.join(r, source_f), plugin="tifffile")
    train = np.concatenate((source, image_dil), axis=-1)
    io.imsave(os.path.join(r, "edges"+str(i+1)+r".tif"), image_dil, plugin="tifffile", imagej=True)
    io.imsave(os.path.join(targ, "train_for_split"+str(i+1)+r".tif"), np.moveaxis(train, 2, 0),
              plugin="tifffile", imagej=True)


train_Unet(path=r"C:\Users\USER\Documents\studia\zaklad\EC_rainbow\cell_edges", ltrain=4, lgt=3)

def measure_dist(i):
    neighbors = edgelist[i]
    boxes = get_boxes(i)
    _, _, x2, y2 = boxes.max(axis=0)
    x1, y1, _, _ = boxes.min(axis=0)
    crop = relabeled[x1:x2, y1:y2].copy()
    crop_phi = np.ones_like(crop)
    crop_phi[crop == i+1] = 0
    distance_map = skfmm.distance(phi=crop_phi)
    neighbour_params = [[i, x] + list(get_params(distance_map, crop, x)) for x in neighbors]
    return neighbour_params