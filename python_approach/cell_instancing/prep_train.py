import numpy as np
import os
from skimage import io
import random
import re
from split_image import split

r = r"C:\Users\USER\Documents\studia\zaklad\EC_rainbow\cells"
os.chdir(r)

files = os.listdir()
files_img = [x for x in files if re.match("test_image[0-9].tif", x)]
files_mask = [x for x in files if re.match("labels[0-9].tif", x)]
files_pred = [x for x in files if re.match("prediction[0-9].tif", x)]

data_split = []
label_split = []
pred_split = []
for j in range(len(files_img)):
 data = io.imread(os.path.join(r, files_img[j]), plugin="tifffile")
 data = data[:,:,:3]
 mask = io.imread(os.path.join(r, files_mask[j]), plugin="tifffile")
 pred = io.imread(os.path.join(r, files_pred[j]), plugin="tifffile")
 data_split.extend(split(np.moveaxis(data, 2, 0), 560, 120))
 label_split.extend(split(np.moveaxis(mask, 2, 0), 560, 120))
 pred_split.extend(split(np.moveaxis(pred, 2, 0), 560, 120))

data_split = [np.moveaxis(x,0,2) for x in data_split]
label_split = [np.moveaxis(x,0,2) for x in label_split]
pred_split = [np.moveaxis(x,0,2) for x in pred_split]

os.mkdir(".\\test_f\\")
os.mkdir(".\\GT\\")
os.mkdir(".\\train_f\\")


for C in range(3):
 ret = [x[60:-60, 60:-60, C].mean() for x in label_split]
 ret = np.where(np.asarray(ret) != 0)[0]

 data_split_C = [data_split[i] for i in ret]
 label_split_C = [label_split[i][:, :, C] for i in ret]
 pred_split_C = [pred_split[i][:, :, C] for i in ret]

 rand = random.sample(range(len(data_split_C)), len(data_split_C))
 test = 0.2
 test = int(len(rand) // (1 / test))
 train = 0.8
 train = int(len(rand) // (1 / train))

 train = rand[-train:]
 test = rand[:test]
 os.mkdir(".\\test_f\\"+"F"+str(C))
 os.mkdir(".\\GT\\"+"F"+str(C))
 os.mkdir(".\\train_f\\"+"F"+str(C))

 for i in range(len(test)):
  l = test[i]
  io.imsave(fname=".\\GT\\F" + str(C) + "\\test" + str(i) + ".tif", arr=label_split_C[l][24:-24,24:-24,...],
            plugin="tifffile", imagej=True)
  io.imsave(fname=".\\test_f\\F" + str(C) + "\\test" + str(i) + ".tif",
            arr=np.concatenate((data_split_C[l], pred_split_C[l][:, :, None]), -1)[24:-24,24:-24,...],
            plugin="tifffile", imagej=True)

 for i in range(len(train)):
  l = train[i]
  io.imsave(fname=".\\GT\\F" + str(C) + "\\train" + str(i) + ".tif", arr=label_split_C[l],
            plugin="tifffile", imagej=True)
  io.imsave(fname=".\\train_f\\F" + str(C) + "\\train" + str(i) + ".tif",
            arr=np.concatenate((data_split_C[l], pred_split_C[l][:, :, None]), -1),
            plugin="tifffile", imagej=True)





python train_dec_kaggle.py --trainDir C:\Users\USER\Documents\studia\zaklad\EC_rainbow\cells\train_f\F0 ^
 --valDir C:\Users\USER\Documents\studia\zaklad\EC_rainbow\cells\test_f\F0 ^
 --annoDir C:\Users\USER\Documents\studia\zaklad\EC_rainbow\cells\GT\F0 ^
 --num_workers 3 --weightDst dec0 --init_lr 0.00003 --num_epochs 90 --decayEpoch 75 ^
 --batch_size 2


python train_dec_kaggle.py --trainDir C:\Users\USER\Documents\studia\zaklad\EC_rainbow\cells\train_f\F1 ^
 --valDir C:\Users\USER\Documents\studia\zaklad\EC_rainbow\cells\test_f\F1 ^
 --annoDir C:\Users\USER\Documents\studia\zaklad\EC_rainbow\cells\GT\F1 ^
 --num_workers 3 --weightDst dec1 --init_lr 0.00003 --num_epochs 90 --decayEpoch 75


python train_dec_kaggle.py --trainDir C:\Users\USER\Documents\studia\zaklad\EC_rainbow\cells\train_f\F2 ^
 --valDir C:\Users\USER\Documents\studia\zaklad\EC_rainbow\cells\test_f\F2 ^
 --annoDir C:\Users\USER\Documents\studia\zaklad\EC_rainbow\cells\GT\F2 ^
 --num_workers 3 --weightDst dec2 --init_lr 0.00003 --num_epochs 90 --decayEpoch 75






python train_seg_kaggle.py --trainDir C:\Users\USER\Documents\studia\zaklad\EC_rainbow\cells\train_f\F0 ^
 --valDir C:\Users\USER\Documents\studia\zaklad\EC_rainbow\cells\test_f\F0 ^
 --annoDir C:\Users\USER\Documents\studia\zaklad\EC_rainbow\cells\GT\F0 ^
 --num_epochs 100 --num_workers 3 --batch_size 3 --init_lr 0.002 --weightDst seg0 ^
 --dec_weights C:\Users\USER\Documents\studia\zaklad\EC_rainbow\ANCIS-Pytorch\dec0\end_model.pth


python train_seg_kaggle.py --trainDir C:\Users\USER\Documents\studia\zaklad\EC_rainbow\cells\train_f\F1 ^
 --valDir C:\Users\USER\Documents\studia\zaklad\EC_rainbow\cells\test_f\F1 ^
 --annoDir C:\Users\USER\Documents\studia\zaklad\EC_rainbow\cells\GT\F1 ^
 --num_epochs 50 --num_workers 3 --batch_size 2 --init_lr 0.002 --weightDst seg1 ^
 --dec_weights C:\Users\USER\Documents\studia\zaklad\EC_rainbow\ANCIS-Pytorch\dec1\end_model.pth


python train_seg_kaggle.py --trainDir C:\Users\USER\Documents\studia\zaklad\EC_rainbow\cells\train_f\F2 ^
 --valDir C:\Users\USER\Documents\studia\zaklad\EC_rainbow\cells\test_f\F2 ^
 --annoDir C:\Users\USER\Documents\studia\zaklad\EC_rainbow\cells\GT\F2 ^
 --num_epochs 50 --num_workers 3 --batch_size 2 --init_lr 0.002 --weightDst seg2 ^
 --dec_weights C:\Users\USER\Documents\studia\zaklad\EC_rainbow\ANCIS-Pytorch\dec2\end_model.pth





python test_seg_kaggle.py --testDir C:\Users\USER\Documents\studia\zaklad\EC_rainbow\cells\test_f\F0 ^
 --annoDir C:\Users\USER\Documents\studia\zaklad\EC_rainbow\cells\GT\F0 ^
 --dec_weights C:\Users\USER\Documents\studia\zaklad\EC_rainbow\ANCIS-Pytorch\dec0\end_model.pth ^
 --seg_weights C:\Users\USER\Documents\studia\zaklad\EC_rainbow\ANCIS-Pytorch\seg0\end_model.pth ^
 --nms_thresh 0.5

