import os
import matplotlib.pyplot as plt
import sys
from skimage import io, img_as_ubyte, img_as_float32, util
import torch
import numpy as np
from torchvision import transforms as T
from skimage import transform
from run_IS.smooth_tiled_predictions import predict_img_with_smooth_windowing
from run_IS.network import U_Net
from split_image import split, join_img
import tqdm
from time import sleep
import cv2
import gc

def find_newest(path, end=".pkl"):
    files = (os.listdir(path))
    files = [os.path.join(path, x) for x in files if x.endswith(end)]
    mod = max(files, key=os.path.getctime)
    return mod


def gen_gauss(size=256):
    x, y = np.meshgrid(np.linspace(-3, 3, size), np.linspace(-3, 3, size))
    d = np.sqrt(x * x + y * y)
    sigma, mu = 1.0, 0.0
    g = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))
    return g


def weight_img(shape, g_size=256, patch_margins=15):
    fin_shape = list(shape)
    fin = np.zeros(fin_shape, dtype=np.float16)
    g = gen_gauss(g_size)
    p = patch_margins
    g_fit = g[p:-p, p:-p].astype(np.float16)
    val = 256 - (2*p)
    ny = fin_shape[0] // val
    nx = fin_shape[1] // val
    resy = fin_shape[0] % val
    resx = fin_shape[1] % val
    for y in range(ny + 1):
        for x in range(nx + 1):
            sy = y * val
            sx = x * val
            if y == ny:
                if x == nx:
                    fin[sy:, sx:] = g_fit[:resy, :resx]
                else:
                    fin[sy:(sy + val), sx:(sx + val)] = g_fit[:resy, :]
            elif x == nx:
                fin[sy:(sy + val), sx:] = g_fit[:, :resx]
            else:
                fin[sy:(sy + val), sx:(sx + val)] = g_fit[:, :]
    return fin


def rot_f_pred(img, i):
    if i == 0:
        return img
    if i == 1:
        return np.swapaxes(img, 0, 1)
    if i == 2:
        return img[::-1, :, :]
    if i == 3:
        return img[::-1, ::-1, :]


def pred_unet_split(img, unet, out, transf, func):
    g = gen_gauss(size=256)
    gs = [g] * out
    g = np.stack(gs, axis=-1)
    del gs

    if next(unet.parameters()).is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    images = split(np.moveaxis(img, 2, 0), tile_size=256, margin=15)
    images = [np.moveaxis(x, 0, 2) for x in images]

    predictions = []
    for image in tqdm.tqdm(images):
        image = img_as_float32(image)
        image -= image.mean()
        image /= (image.std()*10) + 1e-8
        image += 0.3
        pred_comb = np.zeros(list(image.shape[:2]) + [out])
        image_tens = []
        for i in range(4):
            image_mod = func(image.copy(), i)
            image_tens.append(transf(image_mod.copy()))

        image_tens = torch.stack(image_tens)
        with torch.no_grad():
            pred = unet(image_tens.to(device))
        preds = pred.detach().cpu().numpy()
        for i in range(4):
            pred = np.moveaxis(preds[i, :, :, :], 0, 2)
            pred = func(pred, i)
            pred = pred.astype(np.float16)
            pred_comb += pred
        pred_comb /= 4
        pred_comb *= g
        predictions.append(pred_comb)
    del images

    joined = np.zeros(list(img.shape[:2]) + [out])
    for J in range(3):
        preds_J = [x[:, :, J] for x in predictions]
        joined[:, :, J] = join_img(preds_J, 256, 15, img.shape, preserve_type=True)

    return joined.astype(np.float16)


def pred_f_way(img, unet, out, transf, func, patch=256, margin=15):
    flat = pred_unet_split(img, unet, out, transf, func).astype(np.float32)
    poss = (patch/2)-margin
    poss = int(poss)
    img_shift = util.pad(img, pad_width=[[poss, poss], [poss, poss], [0, 0]], mode="reflect")
    weight = weight_img(img.shape[:2], patch, margin)
    weight += weight_img(img_shift.shape[:2])[poss:-poss, poss:-poss]
    del img
    flat += pred_unet_split(img_shift, unet, out, transf, func)[poss:-poss, poss:-poss, :].astype(np.float32)
    del img_shift
    flat_fin = np.zeros(flat.shape)
    for i in range(flat.shape[-1]):
        flat_fin[..., i] = flat[..., i]/weight
    del weight, flat
    flat_fin = 1/(1 + np.exp(-flat_fin))
    flat_fin = flat_fin.astype(np.float16)
    flat_fin = img_as_ubyte(flat_fin)
    return flat_fin


# vessels_preds
def unet_predict(model, img_array, channels, transf):
    if next(model.parameters()).is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    batch = img_array.shape
    img_array = img_as_float32(img_array)
    img_array -= img_array.mean((1, 2))[:, np.newaxis, np.newaxis, :]
    img_array /= (img_array.std((1, 2)) * 10)[:, np.newaxis, np.newaxis, :] + 1e-8
    img_array += 0.3
    pred = np.zeros(batch)
    rg = 5
    ranges = list(range(0, batch[0], rg))
    for i in ranges:
        stak = torch.stack([transf(x).to(device) for x in img_array[i:i+rg, :, :, :channels]])
        with torch.no_grad():
            res = model(stak)
        res = res.cpu()
        res = res.detach().numpy()
        pred[i:i+res.shape[0], :, :, 0] = res[:, 0, :, :]
    return pred


def smooth_umap_predict_vessels(img, model, out_location, transf):
    shape = img.shape
    test = transform.rescale(img.copy(), 0.5, anti_aliasing=False)
    test = img_as_ubyte(test)
    test = test[:, :, None]
    del img
    pred_smooth = predict_img_with_smooth_windowing(
        test,
        window_size=256,
        subdivisions=2,  # Minimal amount of overlap for windowing. Must be an even number.
        nb_classes=1,
        pred_func=(
            lambda img_batch_subdiv: unet_predict(model, img_batch_subdiv, 1, transf)
        )
    )
    gc.collect()
    torch.cuda.empty_cache()
    del test
    predicted_join = torch.sigmoid(transf(pred_smooth[:, :, 0]))
    predicted_join = predicted_join.detach().numpy()[0, :, :]
    predicted_join = img_as_ubyte(predicted_join)
    del pred_smooth
    gc.collect()
    sleep(10)
    gc.collect()
    print(predicted_join.shape)
    predicted_join = cv2.resize(predicted_join, (shape[1], shape[0]))  # Zamieniło wyiary?!?!?
    io.imsave(fname=out_location, arr=predicted_join, plugin="tifffile", imagej=True)

