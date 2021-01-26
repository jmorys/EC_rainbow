import os
import matplotlib.pyplot as plt
from skimage import io, img_as_ubyte, img_as_float32, segmentation, filters, morphology
import os
import sys
from skimage import io, img_as_ubyte, transform, morphology
import re
import random
import numpy as np
from split_image import split, join_img
from scipy import ndimage as ndi

# test
r = r"C:\Users\USER\Documents\studia\zaklad\EC_rainbow\test_voronoi"
file = r"test_img.tif"
img = io.imread(os.path.join(r, file))

cells_img = img[..., :3]

plt.figure(600)
plt.imshow(img_as_ubyte(cells_img > 123))
plt.show()

gauss_downstream = filters.gaussian(cells_img, multichannel=True, sigma=2).astype(np.float32)
threshed1 = gauss_downstream > 0.5
threshed2 = cells_img > cells_img.max() / 2  # TODO in final version 0.5 | 123
threshed = (threshed1 * 1. + threshed2 * 1.) > 1

gauss_energy = filters.gaussian(cells_img, multichannel=True, sigma=5).astype(np.float32)

vesses = np.zeros(gauss_downstream.shape)
for i in range(3):
    vesses[..., i] = filters.sato(gauss_downstream[..., i], black_ridges=False,
                                  sigmas=range(3, 13, 3), mode="reflect")

vesses_prim = np.zeros(gauss_downstream.shape)
for i in range(3):
    vesses_prim[..., i] = filters.sato(gauss_downstream[..., i], black_ridges=True,
                                       sigmas=(1, 2, 4), mode="reflect")

disc = morphology.disk(4)
maxes = morphology.h_maxima(vesses.astype(np.float16), h=0.05, selem=disc[:, :, np.newaxis])
disc_vis = morphology.disk(10)
vised = morphology.dilation(maxes, disc_vis[:, :, np.newaxis])

energy = -(vesses + gauss_energy - vesses_prim)
segmented_cells = np.zeros(cells_img.shape, dtype=np.uint32)
for i in range(3):
    markers, _ = ndi.label(maxes[..., i])
    segmented_cells[..., i] = segmentation.watershed(energy[..., i], markers, mask=threshed[..., i])

io.imsave(fname=os.path.join(r, "tezd.tif"), arr=np.moveaxis(img_as_float32(segmented_cells), 2, 0),
          plugin="tifffile", imagej=True)

plt.figure(dpi=600)
plt.imshow(img_as_ubyte(threshed.sum(axis=-1) > 1))
plt.show()

# segmented_cells_copy = segmented_cells.copy()
segmented_cells[threshed.sum(axis=-1) > 1, :] = 0

smol_img = segmented_cells.copy()

uni = np.unique(smol_img)

summed = sum([len(np.unique(smol_img[..., i])) for i in range(3)])
max_unique = smol_img.max()

unique_cells = (smol_img.argmax(axis=-1) * max_unique) + smol_img.max(axis=-1)

vessels = img[..., 3]
vessels = vessels < 123

masked_cells = unique_cells.copy()
masked_cells[vessels] = 0


relabeled = masked_cells.copy()
for new, old in enumerate(np.unique(masked_cells)):
    relabeled[relabeled == old] = new
masked_cells = np.ma.MaskedArray(masked_cells, vessels)

thisphi = np.ones_like(unique_cells)
thisphi[relabeled > 0] = 0
thisphi = np.ma.MaskedArray(thisphi, vessels)

import skfmm
distances = skfmm.distance(phi=thisphi)

plt.figure(dpi=600)
plt.imshow(img_as_ubyte(distances / distances.max()))
plt.show()

division = segmentation.watershed(distances, relabeled, mask=vessels * (-1) + 1)
division[vessels] = 0
plt.figure(dpi=600)
plt.imshow(img_as_ubyte(division / division.max()))
plt.show()

edges = filters.sobel(division)
from skimage.future import graph
edges = (edges > 0).astype(np.int32)

g = graph.rag_boundary(division, edges * 1., connectivity=4)
g.remove_node(0)

fig, ax = plt.subplots(dpi=600)
lc = graph.show_rag(division, g, threshed, img_cmap=None, edge_cmap='viridis',
                    edge_width=0.5, ax=ax)
fig.colorbar(lc, fraction=0.03)
fig.show()
nodes = list(np.sort(list(g.nodes)))
adjlist = [[x for x in g.neighbors(node)] for node in nodes]
from skimage.measure import regionprops
regions = regionprops(division)


relabeled = np.ma.MaskedArray(relabeled, vessels)



from joblib import Parallel, delayed


def get_params(dist_map, label_map, label):
    vals = dist_map[label_map == label]
    return vals.min(), vals.mean(), vals.max(), vals.std()


class DistanceMeasurer:
    def __init__(self, labeled, adjlist, division_regions):
        self.labeled = labeled
        self.adjlist = adjlist
        self.regions = division_regions

    def __len__(self):
        return len(edgelist)

    def get_boxes(self, nodes):
        boxes = [self.regions[node-1].bbox for node in nodes]
        boxes = np.asarray(boxes)
        _, _, y2, x2 = boxes.max(axis=0)
        y1, x1, _, _ = boxes.min(axis=0)
        return y1, x1, y2, x2

    def __call__(self, i):
        nodes = [regions[i].label]
        nodes.extend(self.edgelist[i])
        y1, x1, y2, x2 = self.get_boxes(nodes)
        crop = self.labeled[y1:y2, x1:x2].copy()
        crop_phi = np.ones_like(crop)
        crop_phi[crop == nodes[0]] = 0
        dist_map = skfmm.distance(crop_phi)
        params = [[nodes[0], label, get_params(dist_map, crop, label)]for label in nodes[1:]]
        return params


measure_dist = DistanceMeasurer(relabeled, adjlist, regions)

from itertools import chain
import multiprocessing as mp


start = time.time()
comb_params = [measure_dist(i) for i in range(len(edgelist))]
edge_params = chain.from_iterable(comb_params)
edge_params = np.asarray(list(edge_params))
end = time.time()
print(end-start)



pool = mp.Pool(mp.cpu_count())
start = time.time()
results = pool.map_async(measure_dist, range(len(edgelist)))
mid = time.time()
edge_params = chain.from_iterable(results.get())
edge_params = np.asarray(list(edge_params))
pool.close()
end = time.time()
print(end-start)
print(end-mid)


labeled_regions = regionprops(relabeled)
color_map = threshed.argmax(axis=-1)
color_map[threshed.max(axis=-1)] += 1
# [nr, col, y, x, area, eccentricity]
results = np.zeros((len(labeled_regions), 6))
for i, region in enumerate(labeled_regions):
    crop = color_map[region.slice].copy()
    col = crop[region.image].mean()
    res = [i+1, col] + list(region.centroid) + [region.area, region.eccentricity]
    results[i, :] = res

np.savetxt(os.path.join(r, "cell_parametres.csv"), results, delimiter=",")
