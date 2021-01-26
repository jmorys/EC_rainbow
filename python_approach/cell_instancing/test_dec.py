from skimage import io
from seg_utils import *
from dec_utils import dec_transforms, dec_dataset_kaggle, Detect, Anchors
from dec_utils.dec_eval import parse_rec
from matplotlib import pyplot as plt
from models import dec_net
import os
import torchvision.transforms as T
import torch.nn.functional as F
import torch
from skimage import img_as_ubyte
from skimage.measure import regionprops
import matplotlib.patches as mpatches
from matplotlib import cm
from eval.universal import voc_ap
import tqdm


def voc_eval(dset, model, detector, ovthresh=0.5,
             use_07_metric=True):
    GT_boxes = []
    all_decs = []
    anchorGen = Anchors(512, 512)
    anchors = anchorGen.forward()
    for index, vals in enumerate(dset):
        data, bbox, label = vals
        bbox = bbox.detach().numpy()
        with torch.no_grad():
            locs, conf = model(data.unsqueeze(0).to(device))
        detections_np = detector(locs, conf, anchors).detach().numpy()[0, 1, :, :]
        detections_np = detections_np.copy()[detections_np[:, 0] > 0, :]
        detections_np = np.concatenate((detections_np, np.ones((detections_np.shape[0], 1)) * index), 1)
        all_decs.append(detections_np)
        GT_boxes.append(bbox)

    all_decs = np.concatenate(all_decs, axis=0)

    sequence = np.argsort(-all_decs[:, 0])
    all_decs = all_decs[sequence, :]

    decs = [np.zeros(bbox.shape[0], dtype="bool") for bbox in GT_boxes]
    tp = np.zeros(all_decs.shape[0])
    fp = np.zeros(all_decs.shape[0])
    for i in range(all_decs.shape[0]):
        bb = all_decs[i, 1:-1]
        index = int(all_decs[i, -1])
        bbox = GT_boxes[index]
        iymin = np.maximum(bbox[:, 0], bb[0])
        ixmin = np.maximum(bbox[:, 1], bb[1])
        iymax = np.minimum(bbox[:, 2], bb[2])
        ixmax = np.minimum(bbox[:, 3], bb[3])
        iw = np.maximum(ixmax - ixmin, 0.)
        ih = np.maximum(iymax - iymin, 0.)
        inters = iw * ih
        union = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                 (bbox[:, 2] - bbox[:, 0]) *
                 (bbox[:, 3] - bbox[:, 1]) - inters)
        overlaps = inters / union
        ovmax = np.max(overlaps)
        jmax = np.argmax(overlaps)
        if ovmax > ovthresh:
            if not decs[index][jmax]:
                decs[index][jmax] = 1
                tp[i] = 1.
            else:
                fp[i] = 1.

    npos = sum([x.shape[0] for x in GT_boxes])
    fp_val = tp.shape[0] - tp.sum()
    tp_val = tp.sum()
    fp = 1 - tp

    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    rec = tp / npos
    ap = voc_ap(rec, prec, use_07_metric)
    return rec, prec, ap


def load_dec_weights(dec_model, dec_weights):
    print('Resuming detection weights from {} ...'.format(dec_weights))
    dec_dict = torch.load(dec_weights)
    dec_dict_update = {}
    for k in dec_dict:
        if k.startswith('module') and not k.startswith('module_list'):
            dec_dict_update[k[7:]] = dec_dict[k]
        else:
            dec_dict_update[k] = dec_dict[k]
    dec_model.load_state_dict(dec_dict_update, strict=True)
    return dec_model


num_classes = 2
dec_weights = r"C:\Users\USER\Documents\studia\zaklad\EC_rainbow\ANCIS-Pytorch\dec0\end_model.pth"

# -----------------load detection model -------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dec_model = dec_net.resnetssd50(pretrained=False, num_classes=num_classes)
dec_model = load_dec_weights(dec_model, dec_weights)
dec_model = dec_model.to(device)
dec_model.eval()

r = r"C:\Users\USER\Documents\studia\zaklad\EC_rainbow\cells"
annoDir = os.path.join(r, r"GT\F0")
valDir = os.path.join(r, r"test_f\F0")

data_transforms = dec_transforms.Compose([dec_transforms.ToTensor()])

dset = dec_dataset_kaggle.NucleiCell(valDir, annoDir, data_transforms,
                                     imgSuffix=".tif", annoSuffix=".tif",
                                     trainb=False)


conf_Ts = np.arange(0.1, 1, 0.1)
nms_Ts = np.arange(0.1, 1, 0.1)
var1 = np.arange(0.1, 0.6, 0.1)
var2 = np.arange(0.1, 0.6, 0.1)

pars = np.array(np.meshgrid(conf_Ts, nms_Ts, var1, var2)).T.reshape(-1, 4)
vals = np.asarray(range(pars.shape[0]))
np.random.shuffle(vals)
test_pars = pars[vals[:pars.shape[0]//2], :]

aps = np.zeros((test_pars.shape[0], 1))
for i in tqdm.trange(test_pars.shape[0]):
    params = test_pars[i, :]
    detector = Detect(num_classes=num_classes,
                      top_k=200,
                      conf_thresh=params[0],
                      nms_thresh=params[1],
                      variance=[params[2], params[3]], soft=False)
    _, _, ap = voc_eval(dset, dec_model, detector, ovthresh=0.5)
    aps[i] = ap

par_data = np.concatenate((test_pars, aps), axis=1)
par_data = par_data[np.argsort(-par_data[:, -1]), :]

unique_pars = np.unique(par_data[:, 2])
valz = np.zeros(unique_pars.shape)
for index, val in enumerate(unique_pars):
    inds = par_data[:, 0] == val
    valz[index] = par_data[inds, -1].max()

plt.plot(unique_pars, valz)
plt.show()


detector = Detect(num_classes=num_classes,
                  top_k=200,
                  conf_thresh=0.1,
                  nms_thresh=0.1,
                  variance=[0.1, 0.1], soft=True)
_, _, ap = voc_eval(dset, dec_model, detector, ovthresh=0.7)

anchorGen = Anchors(512, 512)
anchors = anchorGen.forward()

data, bbox, label = dset[12]


image = np.moveaxis(data.detach().numpy()[:3, ...], 0, 2)

plt.imshow(image[:, :, 0])
plt.show()

label_image = data.detach().numpy()[3, ...]

bbox = bbox.detach().numpy()
with torch.no_grad():
    locs, conf = dec_model(data.unsqueeze(0).to(device))
detections_np = detector(locs, conf, anchors).detach().numpy()[0, 1, :, :]
detections_np = detections_np.copy()[detections_np[:, 0] > 0, :]
boxes_after_nms = detections_np[:, 1:].copy()
conf = detections_np[:, 0]



magma = cm.get_cmap('magma', 256)

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
ax.imshow(label_image, cmap='jet')
for i in range(boxes_after_nms.shape[0]):
    # draw rectangle around segmented coins
    minr, minc, maxr, maxc = boxes_after_nms[i, ...]
    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                              fill=False, edgecolor=magma(conf[i]/conf.max())[:3], linewidth=2)
    ax.add_patch(rect)
plt.show()

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
ax.imshow(label_image, cmap='jet')

for i in range(bbox.shape[0]):
    # draw rectangle around segmented coins
    minr, minc, maxr, maxc = bbox[i, ...]
    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                              fill=False, edgecolor='red', linewidth=2)
    ax.add_patch(rect)
plt.show()

label = r"GT\F0\test2.tif"
label = io.imread(os.path.join(r, label), plugin="tifffile")

io.imshow(label)
io.show()
