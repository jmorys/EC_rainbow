from skimage import io
from seg_utils import *
from dec_utils import dec_transforms, dec_dataset_kaggle, Detect, Anchors
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

data, bbox, label = dset[1]

with torch.no_grad():
    locs, conf = dec_model(data.unsqueeze(0).to(device))

h, w, c = data.shape

detector = Detect(num_classes=num_classes,
                  top_k=200,
                  conf_thresh=0.2,
                  nms_thresh=0.2,
                  variance=[0.1, 0.2])

anchorGen = Anchors(512, 512)
anchors = anchorGen.forward()

detections = detector(locs, conf, anchors)
detections_np = detections.detach().numpy()

pos = np.where(np.sum(detections_np[0, 1, :, :], -1) != 0)
boxes_after_nms = detections_np[0, 1, pos, 1:]
boxes_after_nms = boxes_after_nms[0, ...]
conf = detections_np[0, 1, pos, 0]
conf = conf[0, :]

image = np.moveaxis(data.detach().numpy()[:3, ...], 0, 2)

plt.imshow(image[:, :, 0])
plt.show()

label_image = data.detach().numpy()[3, ...]




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

bbox = bbox.detach().numpy()
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
ax.imshow(label_image, cmap='jet')

for i in range(bbox.shape[0]):
    # draw rectangle around segmented coins
    minr, minc, maxr, maxc = bbox[i, ...]
    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                              fill=False, edgecolor='red', linewidth=2)
    ax.add_patch(rect)
plt.show()

label = r"GT\F0\test1.tif"
label = io.imread(os.path.join(r, label), plugin="tifffile")


semantic = data[3, ...].detach().numpy()

io.imshow(label)
io.show()
io.imshow(semantic)
io.show()

var = (bbox[:, 0] - bbox[:, 2]) * (bbox[:, 1] - bbox[:, 3])
