import matplotlib.pyplot as plt
import numpy as np
import skfmm
from scipy import ndimage as ndi
from skimage import filters
from skimage import io, morphology, transform
from skimage.feature import peak_local_max
from skimage.filters import gaussian
from skimage.segmentation import watershed

r = r"C:\Users\USER\Documents\studia\zaklad\EC_rainbow\10092020RAINBOW EXP ii"
r = r + "\\Rainbow_d7_AE0091_2_final_ zeiss_Maximum intensity projection"
r = r + "\\"

image = io.imread(r + "vessel_mask.tif", plugin="tifffile")
image = image[0, :, :]
image = transform.rescale(image, (2, 2), anti_aliasing=True)
phi = np.ones_like(image, dtype=np.int8)
phi = np.ma.MaskedArray(phi, (image < 0.5))
del image

i = 3
image = io.imread(r + "\\split\\C" + str(i) + "\\result_RAW.tif", plugin="tifffile")
image = gaussian((image*225).astype("uint8"), sigma=2, multichannel=True)
image = image.astype(np.float16)
image = (image*225).astype("uint8")

thresholds = filters.threshold_multiotsu(image, classes=2)
regions = np.digitize(image, bins=thresholds)
regions = morphology.remove_small_objects(regions > 0, min_size=400)

# distance = ndi.distance_transform_edt(regions)
# local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)),
#                             labels=regions, min_distance=60)
# markers = ndi.label(local_maxi)[0]
# labels = watershed(-distance, markers, mask=regions)
labels = ndi.label(regions)[0]
# del markers, local_maxi, regions
labels = morphology.remove_small_objects(labels, min_size=200)


# BIG BRAIN
smol_phi = phi[4000:6000, 3000:5000]
smol_img = labels[4000:6000, 3000:5000]
mask = smol_phi.mask
smol_img[mask] = 0
plt.figure(dpi=800)
plt.imshow(smol_img)
plt.show()
plt.figure(dpi=800)
plt.imshow(np.ma.MaskedArray(smol_img, mask))
plt.show()


uni = np.unique(smol_img)

voro_tensor = np.zeros(shape=(len(uni), smol_img.shape[0], smol_img.shape[1]))
voro_tensor[0, :, :] = 10000
voro_tensor = np.ma.MaskedArray(voro_tensor, np.tile(mask, (len(uni), 1, 1)))

for i in range(1, len(uni)):
    thisphi = smol_phi.copy()
    thisphi[smol_img == uni[i]] = 0
    voro_tensor[i, :, :] = skfmm.distance(phi=thisphi)
    del thisphi


flat = np.nanargmin(voro_tensor, axis=0)

flat = np.ma.MaskedArray(flat, mask)

plt.figure(dpi=800)
plt.imshow(flat)
io.show()


edges = filters.sobel(flat)
edges_rgb = color.gray2rgb(edges)
from skimage.future import graph
plt.figure(dpi=800)
g = graph.rag_boundary(flat, edges, connectivity=1)
g.remove_node(0)

lc = graph.show_rag(flat, g, edges_rgb, img_cmap=None, edge_cmap='viridis',
                    edge_width=1.2)

plt.colorbar(lc, fraction=0.03)
plt.show()

for i in range(1, len(uni)):
    print([uni[x] for x in g.neighbors(i)])

# plt.figure(dpi=500)
# plt.scatter(cols[rpos, 0], cols[rpos,1])
# plt.grid(True)
# plt.show()






phi[tuple(pos)] = 0
speed = np.ones_like(phi)
speed = np.ma.MaskedArray(speed, (smol_img == 0))
dist = skfmm.distance(phi=phi)


plt.figure(dpi=800)
plt.imshow(dist)
plt.show()


