import matplotlib.pyplot as plt
from skimage import io, img_as_ubyte, img_as_float32, segmentation, filters, morphology
import os
import sys
import numpy as np
from split_image import split, join_img
from scipy import ndimage as ndi
from python_approach.watershed_tools import *
import tqdm
import skfmm

# test
r = r"C:\Users\USER\Documents\studia\zaklad\EC_rainbow\test_voronoi"
file = r"test_img.tif"
img = io.imread(os.path.join(r, file))

sato1 = CreateSato(range(4, 17, 4))
sato2 = CreateSato([1, 3])
relabeled, cell_params, edge_params = get_graph(img, sato1=sato1, sato2=sato2)









#
#
# # test_AP
# # optimal : m1=0.6, m2=0.6, a=1, b=0.35, c=3
#
#
# def evaluation(vals):
#     prec = (vals[..., 0] / (vals[..., 0] + vals[..., 1]))
#     prec = prec.min(axis=0).sum()
#     rec = (vals[..., 0] / vals[..., 2]).sum()
#     rec= rec.min(axis=0).sum()
#     return rec, prec
#
# sato1 = CreateSato(range(4, 17, 4))
# sato2 = CreateSato([1, 3])
#
# a = np.linspace(1.5, 3, 4)
# c = np.linspace(1, 3, 4)
# grid = np.meshgrid(a, c)
# grid = np.asarray(grid).T
# grid = grid.reshape(-1, 2)
#
# comb = []
# for i, vals in tqdm.tqdm(enumerate(grid)):
#     params = [vals[0]] + [0.35] + [vals[1]] + [0.6, 0.6]
#     vals = image_recall(sato1, sato2, pred_folder=r"C:\Users\USER\Documents\studia\zaklad\EC_rainbow\test_voronoi",
#                  GT_folder=r"C:\Users\USER\Documents\studia\zaklad\EC_rainbow\cells",
#                  params=params)
#     comb.append(vals)
#
# evaluations = [evaluation(vals)[0] for vals in comb]
# evaluations = np.asarray(evaluations)
# np.asarray(evaluations).argmax()
# plt.scatter(grid[:, 0], grid[:, 1], c=evaluations)
# plt.show()
#
#
# besties_a = [evaluations[np.where(grid[:, 0] == val)[0]].max() for val in a]
# besties_b = [evaluations[np.where(grid[:, 1] == val)[0]].max() for val in b]
# besties_c = [evaluations[np.where(grid[:, 2] == val)[0]].max() for val in c]
# plt.plot(c, besties_c)
# plt.show()
#
#
# results_sum = comb[11]
# prec = (results_sum[..., 0] / (results_sum[..., 0] + results_sum[..., 1]))
# rec = (results_sum[..., 0] / results_sum[..., 2])
# overlap_thresholds = np.linspace(0.1, 0.9, 9)
# plt.plot(overlap_thresholds, rec[0, :], label="1")
# plt.plot(overlap_thresholds, rec[1, :], label="2")
# plt.plot(overlap_thresholds, rec[2, :], label="3")
# plt.legend()
# plt.show()
#
# plt.plot(overlap_thresholds, prec[0, :], label="1")
# plt.plot(overlap_thresholds, prec[1, :], label="2")
# plt.plot(overlap_thresholds, prec[2, :], label="3")
# plt.legend()
# plt.show()


