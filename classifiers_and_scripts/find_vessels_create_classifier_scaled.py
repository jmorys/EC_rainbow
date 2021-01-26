from ij import IJ, ImageStack
from ij.io import DirectoryChooser
from ij.plugin import ChannelSplitter
from trainableSegmentation import WekaSegmentation, FeatureStackArray, FeatureStack
from weka.classifiers.functions import SMO

from net.haesleinhuepf.clij2 import CLIJ2
clij2 = CLIJ2.getInstance()
# vessel detection training


dc = DirectoryChooser("Choose a folder")
folder = dc.getDirectory()
pos = IJ.openImage(folder + "\\vessels_positive_paint.tif")
pos.show()
pos.getProcessor().threshold(125)
IJ.run(pos, "Create Selection", "")
pos.updateAndDraw()
posroi = pos.getRoi()


neg = IJ.openImage(folder + "\\vessels_negative_paint.tif")
neg.show()
neg.getProcessor().threshold(125)
IJ.run(pos, "Create Selection", "")
neg.updateAndDraw()
negroi = pos.getRoi()


neg.close()
pos.close()

img = IJ.openImage(folder + "\\train_img.tif")

channels = ChannelSplitter.split(img)
img.close()
image = channels[3]
channels = channels[0:3]
proc = image.getProcessor()

# run entropies
print("entropies start")
entropies = range(2, 20, 6)
entropies_imgs = []
src = clij2.push(image)
dst = clij2.create(src)
for entropy in entropies:
    clij2.entropyBox(src, dst, entropy, entropy, 1)
    result = clij2.pull(dst)
    entropies_imgs.append(result)

# run gaussians
print("gaussian start")
gaussians = range(2, 40, 6)
gaussians_imgs = []
for gaussian in gaussians:
    clij2.gaussianBlur2D(src, dst, gaussian, gaussian)
    result = clij2.pull(dst)
    gaussians_imgs.append(result)

#clear memory
src.close()
dst.close()
clij2.clear()

# construct feature stack
featuresArray = FeatureStackArray(image.getStackSize())
stack = ImageStack(image.getWidth(), image.getHeight())
for i in range(len(entropies)):
    stack.addSlice("entropy"+str(entropies[i]), entropies_imgs[i].getProcessor())

for i in range(len(gaussians)):
    stack.addSlice("gaussian"+str(gaussians[i]), gaussians_imgs[i].getProcessor())
#free memory
del gaussians_imgs
del entropies_imgs

# put feature stack into array
features = FeatureStack(stack.getWidth(), stack.getHeight(), False)
features.setStack(stack)
featuresArray.set(features, 0)
featuresArray.setEnabledFeatures(features.getEnabledFeatures())

smo = SMO()

wekaSegmentation = WekaSegmentation(image)
wekaSegmentation.setFeatureStackArray(featuresArray)
wekaSegmentation.setClassifier(smo)

wekaSegmentation.addExample(0, posroi, 1)
wekaSegmentation.addExample(1, negroi, 1)
wekaSegmentation.trainClassifier()
wekaSegmentation.saveClassifier(folder + "\\vessel-classifier_scaled.model")
output = wekaSegmentation.applyClassifier(image, featuresArray, 0, True)

output.show()
