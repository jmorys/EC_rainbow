import argparse
import torch.optim as optim
from torch.optim import lr_scheduler
from skimage import io, img_as_float, transform
from seg_utils import *
from dec_utils import *
from seg_utils import seg_transforms, seg_dataset_kaggle, seg_eval
from matplotlib import pyplot as plt
from models import dec_net_seg, seg_net
import cv2
import os


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
seg_weights = r"C:\Users\USER\Documents\studia\zaklad\EC_rainbow\ANCIS-Pytorch\seg0\end_model.pth"

# -----------------load detection model -------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dec_model = dec_net_seg.resnetssd50(pretrained=False, num_classes=num_classes)
dec_model = load_dec_weights(dec_model, dec_weights)
dec_model = dec_model.to(device)
dec_model.eval()
# -----------------load segmentation model -------------------------
seg_model = seg_net.SEG_NET(num_classes=num_classes)
seg_model.load_state_dict(torch.load(seg_weights))
seg_model = seg_model.to(device)
seg_model.eval()


r = r"C:\Users\USER\Documents\studia\zaklad\EC_rainbow\cells\test_f\F0"

data = io.imread(os.path.join(r, "test1.tif"), plugin="tifffile")
image_o = img_as_float(data)
rimg = image_o[..., :3].copy()
rimg -= rimg.mean()
rimg /= (rimg.std() * 10) + 1e-8
rimg += 0.3
image = np.concatenate((rimg, image_o[..., 3].copy()[..., np.newaxis]), axis=-1)
image = image.astype(np.float32)
image = transform.rotate(image, 45)
h, w, c = data.shape

detector = Detect(num_classes=num_classes,
                  top_k=500,
                  conf_thresh=0.2,
                  nms_thresh=0.2,
                  variance=[0.1, 0.2])

anchorGen = Anchors(512, 512)
anchors = anchorGen.forward()

imggs = []
imggs.append(image.copy())
imggs.append(image.copy()[:, ::-1, :])
imggs.append(image.copy()[::-1, :, :])
imggs.append(image.copy()[::-1, ::-1, :])

predictions = []
for imgg in imggs:
    x = torch.from_numpy(imgg.copy().transpose((2, 0, 1)))
    x = x.unsqueeze(0)
    x = x.to(device)
    locs, conf, feat_seg = dec_model(x)
    del(x)
    detections = detector(locs, conf, anchors)
    outputs = seg_model(detections, feat_seg)
    mask_patches, mask_dets = outputs

    ori_img = imgg.copy()

    for b_mask_patches, b_mask_dets in zip(mask_patches, mask_dets):
        nd = len(b_mask_dets)
        # Step1: rearrange mask_patches and mask_dets
        for d in range(nd):
            d_mask = np.zeros((512, 512), dtype=np.float32)
            d_mask_det = b_mask_dets[d].data.cpu().numpy()
            d_mask_patch = b_mask_patches[d].data.cpu().numpy()
            d_bbox = d_mask_det[0:4]
            d_conf = d_mask_det[4]
            d_class = d_mask_det[5]
            if d_conf < 0.3:
                continue
            [y1, x1, y2, x2] = d_bbox
            y1 = np.maximum(0, np.int32(np.round(y1)))
            x1 = np.maximum(0, np.int32(np.round(x1)))
            y2 = np.minimum(np.int32(np.round(y2)), 512 - 1)
            x2 = np.minimum(np.int32(np.round(x2)), 512 - 1)
            d_mask_patch = cv2.resize(d_mask_patch, (x2 - x1 + 1, y2 - y1 + 1))
            d_mask_patch = np.where(d_mask_patch >= 0.5, 1., 0.)
            d_mask[y1:y2 + 1, x1:x2 + 1] = d_mask_patch
            d_mask = cv2.resize(d_mask, dsize=(w, h), interpolation=cv2.INTER_NEAREST)
            ori_img = map_mask_to_image(d_mask, ori_img[..., :3], color=np.random.rand(3))
    predictions.append(ori_img)

predictions[1] = predictions[1][:, ::-1, :]
predictions[2] = predictions[2][::-1, :, :]
predictions[3] = predictions[3][::-1, ::-1, :]

torch.cuda.empty_cache()
for image in predictions:
    plt.figure(dpi=500)
    plt.imshow(image)
    plt.show()

