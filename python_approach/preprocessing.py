import os
import matplotlib.pyplot as plt
import sys
from skimage import io
import numpy as np
from random import sample
from sklearn import decomposition as decom
from scipy.signal import find_peaks
import math
from skimage.filters import gaussian

sys.path.append(os.path.abspath(r"C:\Users\USER\Documents\studia\zaklad\EC_rainbow\python_approach"))
from split_image import split, join_img


def euq_dist_z(arr):
    return np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2)


def elum(Z):
    if Z < 0:
        return np.exp(Z)
    else:
        return Z


elum = np.vectorize(elum)


def uncorrelate(img):
    channels = np.reshape(img, (img.shape[0] * img.shape[1], img.shape[2]))
    print(channels.mean())
    if channels.shape[0] > 200000:
        rpos = sample(range(channels.shape[0]), 200000)
    else:
        rpos = range(channels.shape[0])

    samp = channels[rpos, :]
    pca = decom.PCA(n_components=3)
    pca = pca.fit(samp)
    pc = pca.transform(channels)
    pc = pc.astype(np.float32)
    del channels
    # plt.figure(dpi=800)
    # plt.scatter(pc[rpos, 2], pc[rpos, 1])
    # plt.show()

    polar = np.zeros((len(rpos), 3))
    polar[:, 0] = euq_dist_z(pc[rpos, 1:])
    polar[:, 1] = np.arctan2(pc[rpos, 1], pc[rpos, 2])

    r = [(x * math.pi / 180) - math.pi for x in range(360)]

    bins = np.digitize(polar[:, 1], bins=r)

    vals = []
    for b in range(len(r)):
        bin_arr = polar[np.where(bins == b), 0]
        try:
            max = np.percentile(bin_arr, 98)
        except:
            max = 0
        vals.append(max)

    peaks = []
    p = 1000
    i = 0
    lr = 0.1
    while len(peaks) != 3:

        if len(peaks) > 3:
            peaks, _ = find_peaks(vals, prominence=p)
            p = p * (1+lr)
        if len(peaks) < 3:
            p = p * (1 - lr)
            peaks, _ = find_peaks(vals, prominence=p)

        i = i+1
        if (i % 100) == 0:
            lr = lr*0.9

        if i >= 10000:
            print("Couldn't converge")
            return None

    print("peaks:")
    print(np.array(r)[peaks])

    cols = np.zeros(shape=pc.shape, dtype=np.float32)
    for i in range(len(peaks)):
        theta = r[peaks[i]]
        rot = np.array(((np.cos(theta), -np.sin(theta)),
                        (np.sin(theta), np.cos(theta))))
        rotated = rot.dot(pc[:, 1:3].transpose())
        sd = rotated[0, :].std() * 2
        cols[:, i] = elum(rotated[1, :]) * np.sqrt(elum(-(rotated[0, :] - 3 * sd) * (rotated[0, :] + 3 * sd)))
    print(cols.mean())
    cols = np.nan_to_num(cols)
    return cols


r = r"C:\Users\USER\Documents\studia\zaklad\EC_rainbow\cells"

image = io.imread(r + r"\prep_next.tif", plugin="tifffile")
image = image[:, :, 0:3]
sh = image.shape
image = gaussian(image, sigma=3, multichannel=True)
image = image.astype(np.float16)
uncorrelated = uncorrelate(image*2**16)
uncorrelated = uncorrelated.reshape(sh)
uncorrelated = np.sqrt(uncorrelated)

io.imsave(fname=r + r"\pca_correction_next.tif", arr=uncorrelated.astype("uint16"))

del uncorrelated










i = 1
while True:
    i+=1

    if (i%100) == 0:
        print(i)
    if i >= 10000:
        print("done")
        break



channels = np.reshape(image, (sh[0] * sh[1], sh[2]))
del image
channels = channels.astype(np.float16)


rand = sample(range(channels.shape[0]), 10000)

plt.figure(dpi=800)
plt.scatter(X_transformed[rand, 2], X_transformed[rand, 1])
plt.show()


plt.figure(dpi=400)
histogram = np.histogram(X_transformed[:, 2], bins=1000)
plt.plot( histogram[1][:-1], histogram[0])
plt.show()



splet = split(image, 512, 0)

unc =[]
for i in range(len(splet)):
    try:
        val = uncorrelate(splet[i])
    except:
        val = uncorrelate(splet[i])
    unc.append(val)

unc_cop = unc[:]
unc = [x.reshape(splet[0].shape) for x in unc]

imagee = join_img([x[:,:,2] for x in unc], 512, 1, image.shape)





#
# col = cols[:, i]
# plt.figure(dpi=500)
# plt.scatter(rotated[1, rpos], col[rpos])
# plt.grid(True)
# plt.show()

# plt.figure(dpi=500)
# plt.scatter(cols[rpos, 0], cols[rpos,1])
# plt.grid(True)
# plt.show()
#
# plt.figure(dpi=800)
# plt.imshow(np.reshape(np.nan_to_num(cols[:, 2]).astype(np.float64), (sh[0], sh[1])))
# plt.show()

plt.figure(dpi=800)
plt.imshow(image[:,:,2])
plt.show()

plt.figure(dpi=800)
plt.imshow(uncorrelated[:,:,2])
plt.show()


for i in range(3):
    plt.figure(dpi=800)
    plt.imshow(np.reshape(cols[:, i].astype(np.float64), (sh[0], sh[1])))
    plt.show()


hist = np.histogram2d(pc[:, 1], pc[:, 2], bins=1000)
hist = hist[0].T


