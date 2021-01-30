import os
import matplotlib.pyplot as plt
from skimage import io, img_as_float32, segmentation, filters, morphology, measure
from skimage.future import graph
import sys
import re
import random
import numpy as np
from split_image import split, join_img
from scipy import ndimage as ndi
import numpy as np
import skfmm
from itertools import chain
import multiprocessing as mp
from skimage.measure import regionprops


def get_params(dist_map, label_map, label):
    vals = dist_map[label_map == label]
    return vals.min(), vals.mean(), vals.max(), vals.std()


class DistanceMeasurer:
    def __init__(self, labeled, adjlist, division_regions):
        self.labeled = labeled
        self.adjlist = adjlist
        self.regions = division_regions

    def __len__(self):
        return len(adjlist)

    def get_boxes(self, nodes):
        boxes = [self.regions[node - 1].bbox for node in nodes]
        boxes = np.asarray(boxes)
        _, _, y2, x2 = boxes.max(axis=0)
        y1, x1, _, _ = boxes.min(axis=0)
        return y1, x1, y2, x2

    def __call__(self, i):
        nodes = [self.regions[i].label]
        nodes.extend(self.adjlist[i])
        y1, x1, y2, x2 = self.get_boxes(nodes)
        crop = self.labeled[y1:y2, x1:x2].copy()
        crop_phi = np.ones_like(crop)
        crop_phi[crop == nodes[0]] = 0
        dist_map = skfmm.distance(crop_phi)
        params = [[nodes[0], label, get_params(dist_map, crop, label)] for label in nodes[1:]]
        return params


class CreateSato:
    def __init__(self, sigmas, black_ridges=False):
        self.sigmas = list(sigmas)
        self.black = black_ridges

    def __call__(self, img):
        if len(img.shape) == 3:
            layers = img.shape[2]
        else:
            img = img[..., np.newaxis]
            layers = 1
        sato = np.zeros(img.shape)
        for i in range(layers):
            sato[..., i] = filters.sato(img[..., i], black_ridges=self.black,
                                        sigmas=self.sigmas, mode="reflect")
        return sato


def segment_image(cells_img, sato1, sato2, a=1., b=1., c=1., m1=1., m2=1.):
    cells_img = img_as_float32(cells_img)
    gauss_downstream = filters.gaussian(cells_img, multichannel=True, sigma=2).astype(np.float32)
    threshed1 = gauss_downstream > 0.5
    threshed2 = cells_img > 0.5
    threshed = (threshed1 * 1. + threshed2 * 1.) > 0
    del threshed1, threshed2
    gauss_energy = filters.gaussian(cells_img, multichannel=True, sigma=6).astype(np.float32)

    vesses = sato1(gauss_downstream)
    vesses_prim = sato2(gauss_downstream)
    maxima_vals = vesses*m1 + gauss_energy*m2
    disc = morphology.disk(4)
    maxes = morphology.h_maxima(maxima_vals, h=0.05, selem=disc[:, :, np.newaxis])
    del maxima_vals
    energy = -(a*vesses + b*gauss_energy - c*vesses_prim)
    segmented_cells = np.zeros(cells_img.shape, dtype=np.uint32)
    for i in range(3):
        markers, _ = ndi.label(maxes[..., i])
        segmented_cells[..., i] = segmentation.watershed(energy[..., i], markers, mask=threshed[..., i])

    return threshed, segmented_cells


def ft_tp_layer(GT, pred, ovthresh=0.1):
    regions_GT = measure.regionprops(GT)
    regions_pred = measure.regionprops(pred)
    pred_boxes = [pred_reg.bbox for pred_reg in regions_pred]
    BBGT = [GT_reg.bbox for GT_reg in regions_GT]
    BBGT = np.asarray(BBGT)

    tp = np.zeros(len(pred_boxes))
    fp = np.zeros(len(pred_boxes))
    decs = np.zeros(len(regions_GT), dtype=bool)

    for d, bb in enumerate(pred_boxes):
        iymin = np.maximum(BBGT[:, 0], bb[0])
        ixmin = np.maximum(BBGT[:, 1], bb[1])
        iymax = np.minimum(BBGT[:, 2], bb[2])
        ixmax = np.minimum(BBGT[:, 3], bb[3])
        iw = np.maximum(ixmax - ixmin, 0.)
        ih = np.maximum(iymax - iymin, 0.)
        inters = iw * ih
        union = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                 (BBGT[:, 2] - BBGT[:, 0]) *
                 (BBGT[:, 3] - BBGT[:, 1]) - inters)
        overlaps = inters / union
        candidates = overlaps > 0.01
        if sum(candidates) > 0:
            candidates = np.where(candidates)[0]
            overlaps = np.zeros(len(candidates))
            for i, candidate in enumerate(candidates):
                box_stack = np.stack((BBGT[candidate, :], bb), axis=0)
                x1, y1, _, _ = box_stack.min(axis=0)
                _, _, x2, y2 = box_stack.max(axis=0)
                GT_lab = GT[x1:x2, y1:y2]
                pred_lab = pred[x1:x2, y1:y2]
                GT_lab = (GT_lab == regions_GT[candidate].label)
                pred_lab = pred_lab == regions_pred[d].label
                binary_stack = np.stack((GT_lab, pred_lab), axis=-1)
                iou_img = binary_stack.sum(axis=-1)
                overlaps[i] = (iou_img == 2).sum() / (iou_img > 0).sum()
            ovmax = np.max(overlaps)
            jmax = candidates[np.argmax(overlaps)]
        else:
            ovmax = 0
        if ovmax > ovthresh:
            if not decs[jmax]:
                tp[d] = 1.
                decs[jmax] = True
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.

    return fp, tp, len(regions_GT)


def image_recall(sato1, sato2, pred_folder=r"C:\Users\USER\Documents\studia\zaklad\EC_rainbow\test_voronoi",
                 GT_folder=r"C:\Users\USER\Documents\studia\zaklad\EC_rainbow\cells",
                 params=(1, 1, 1, 0.6, 0.6)):
    r = pred_folder
    test_imgs = [os.path.join(r, file) for file in os.listdir(r) if file.startswith("test_image")]
    r = GT_folder
    GT_imgs = [os.path.join(r, file) for file in os.listdir(r) if file.startswith("labels")]

    results_comb = []
    overlap_thresholds = np.linspace(0.1, 0.9, 9)
    for f, files in enumerate(zip(test_imgs, GT_imgs)):
        test_f, GT_f = files
        img = io.imread(test_f, plugin="tifffile")
        threshed, segmented_cells = segment_image(img[..., :3], sato1, sato2, params[0],
                                                  params[1], params[2], params[3], params[4])
        segmented_cells[img[..., 3] < 123, :] = 0
        GT = io.imread(GT_f, plugin="tifffile")
        GT[img[..., 3] < 123, :] = 0
        results = np.zeros((3, len(overlap_thresholds), 3))

        for i in range(3):
            GT_lay = GT[..., i]
            GT_lay = ndi.label(GT_lay)[0]
            segmented_cells_lay = segmented_cells[..., i]
            for t, overlap_thresh in enumerate(overlap_thresholds):
                vals = ft_tp_layer(GT_lay, segmented_cells_lay, ovthresh=overlap_thresh)
                fp_sum = vals[0].sum()
                tp_sum = vals[1].sum()
                results[i, t, :] = [tp_sum, fp_sum, vals[2]]

        results_comb.append(results)
        results_stack = np.stack(results_comb)
        results_sum = results_stack.sum(axis=0)
        return results_sum


def get_graph(img, sato1, sato2, m1=0.6, m2=0.6, a=1., b=0.35, c=3.):
    cells_img = img[..., :3]
    vessels = img_as_float32(img[..., 3])
    vessels = vessels < 0.5
    del img
    # cell segmentation
    threshed, segmented_cells = segment_image(cells_img, sato1, sato2, m1=m1, m2=m2,
                                              a=a, b=b, c=c)
    segmented_cells[threshed.sum(axis=-1) > 1, :] = 0
    max_unique = segmented_cells.max()
    unique_cells = (segmented_cells.argmax(axis=-1) * max_unique) + segmented_cells.max(axis=-1)

    # relabeling
    masked_cells = unique_cells.copy()
    masked_cells[vessels] = 0
    relabeled = masked_cells.copy()
    for new, old in enumerate(np.unique(masked_cells)):
        relabeled[relabeled == old] = new
    del masked_cells

    # divide vessels
    thisphi = np.ones_like(relabeled)
    thisphi[relabeled > 0] = 0
    thisphi = np.ma.MaskedArray(thisphi, vessels)
    distances = skfmm.distance(phi=thisphi)
    division = segmentation.watershed(distances, relabeled, mask=vessels * (-1) + 1)
    division[vessels] = 0


    # create graph
    edges = filters.sobel(division)
    edges = (edges > 0).astype(np.float32)
    g = graph.rag_boundary(division, edges, connectivity=4)
    g.remove_node(0)

    # prepare to measure cell distances
    nodes = list(np.sort(list(g.nodes)))
    adjlist = [[x for x in g.neighbors(node)] for node in nodes]
    regions = regionprops(division)
    relabeled = np.ma.MaskedArray(relabeled, vessels)
    measure_dist = DistanceMeasurer(relabeled, adjlist, regions)

    # measure cell distances
    pool = mp.Pool(mp.cpu_count())
    results = pool.map_async(measure_dist, range(len(adjlist)))
    edge_params = chain.from_iterable(results.get())
    edge_params = np.asarray(list(edge_params))
    pool.close()

    # measure cell properties
    labeled_regions = regionprops(relabeled)
    color_map = threshed.argmax(axis=-1)
    color_map[threshed.max(axis=-1)] += 1
    # [nr, col, y, x, area, eccentricity]
    cell_params = np.zeros((len(labeled_regions), 6))
    for i, region in enumerate(labeled_regions):
        crop = color_map[region.slice].copy()
        col = crop[region.image].mean()
        res = [i + 1, col] + list(region.centroid) + [region.area, region.eccentricity]
        cell_params[i, :] = res

    return relabeled, cell_params, edge_params


