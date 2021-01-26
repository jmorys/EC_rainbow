import numpy as np
from skimage import img_as_ubyte, img_as_float


def split(img, tile_size, margin, preserve_type=True):
    sh = list(img.shape)
    if len(sh) == 2:
        stride = tile_size - 2 * margin
        step = tile_size

        nrows, ncols = (sh[0] // stride)+1, (sh[1] // stride)+1
        sh_ = sh[:]
        sh_[0], sh_[1] = (nrows*stride) + (margin * 2), (ncols*stride) + (margin * 2)

        if preserve_type:
            img_ = np.zeros(shape=sh_, dtype=img.dtype)
        else:
            img_ = np.zeros(shape=sh_)

        img_[margin:(sh[0] + margin), margin:(sh[1] + margin)] = img
        img_[:margin, :] = np.flip(img_[margin:(2 * margin), :], axis=0)
        img_[:, :margin] = np.flip(img_[:, margin:(2*margin)], axis=1)
        y_diff = ((sh[0] + margin) - sh_[0])
        img_[(sh[0] + margin):, :] = np.flip(img_[2 * y_diff:y_diff, :], axis=0)
        x_diff = ((sh[1] + margin)-sh_[1])
        img_[:, (sh[1] + margin):] = np.flip(img_[:, 2*x_diff:x_diff], axis=1)

        splitted = []
        for i in range(nrows):
            for j in range(ncols):
                h_start = j*stride
                v_start = i*stride
                cropped = img_[v_start:v_start+step, h_start:h_start+step]
                splitted.append(cropped)
        return splitted

    if len(sh) == 3:
        stride = tile_size - 2 * margin
        step = tile_size

        nrows, ncols = (sh[1] // stride) + 1, (sh[2] // stride) + 1
        sh_ = sh[:]
        sh_[1], sh_[2] = (nrows * stride) + (margin * 2), (ncols * stride) + (margin * 2)

        if preserve_type:
            img_ = np.zeros(shape=sh_, dtype=img.dtype)
        else:
            img_ = np.zeros(shape=sh_)

        img_[:, margin:(sh[1] + margin), margin:(sh[2] + margin)] = img
        img_[:, :margin, :] = np.flip(img_[:, margin:(2 * margin), :], axis=1)
        img_[:, :, :margin] = np.flip(img_[:, :, margin:(2 * margin)], axis=2)
        y_diff = ((sh[1] + margin) - sh_[1])
        img_[:, (sh[1] + margin):, :] = np.flip(img_[:, 2 * y_diff:y_diff, :], axis=1)
        x_diff = ((sh[2] + margin) - sh_[2])
        img_[:, :, (sh[2] + margin):] = np.flip(img_[:, :, 2 * x_diff:x_diff], axis=2)

        splitted = []
        for i in range(nrows):
            for j in range(ncols):
                h_start = j * stride
                v_start = i * stride
                cropped = img_[:, v_start:v_start + step, h_start:h_start + step]
                splitted.append(cropped)
        return splitted


def join_img(splitted, tile_size, margin, original_shape, preserve_type=True):
    splitted = [x[margin:-margin, margin:-margin] for x in splitted]
    s = range(len(splitted))
    s = np.array(s)
    sh = original_shape
    stride = tile_size - 2 * margin
    nrows, ncols = (sh[0] // stride) + 1, (sh[1] // stride) + 1

    rext = nrows * (tile_size - 2*margin)
    cext = ncols * (tile_size - 2*margin)
    if preserve_type:
        img_full = np.zeros(shape=(rext, cext), dtype=splitted[1].dtype)
    else:
        img_full = np.zeros(shape=(rext, cext))

    rows = s // ncols
    cols = np.mod(s, ncols)
    for i in range(len(splitted)):
        img_full[(rows[i] * stride):((rows[i] + 1) * stride),
        (cols[i] * stride):((cols[i] + 1) * stride)] = splitted[i]

    img = img_full[:sh[0], :sh[1]]
    return img
