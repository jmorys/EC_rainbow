from ij import ImagePlus, IJ, WindowManager, ImageStack
from ij.process import FloatProcessor, ShortProcessor
from ij.io import DirectoryChooser
from ij.plugin import ChannelSplitter, ImageCalculator
from trainableSegmentation import WekaSegmentation, FeatureStackArray, FeatureStack
from java.lang import System
# vessel detection training
from inra.ijpb import morphology
# MorpholibJ package must be activated
from features import TubenessProcessor
import threading

from net.haesleinhuepf.clij2 import CLIJ2
clij2 = CLIJ2.getInstance()

def run_tube(image, sigma, index, result):
    tp = TubenessProcessor(sigma, True)
    tubimg = tp.generateImage(image)
    tubimg = tubimg.getProcessor().convertToShortProcessor()
    tubimg = ImagePlus("img", tubimg)
    result[index] = tubimg


# vessel detection training


dc = DirectoryChooser("Choose a folder")
folder = dc.getDirectory()
pos = IJ.openImage(folder + "\\vessels_positive.tif")
posroi = pos.getRoi()
neg = IJ.openImage(folder + "\\vessels_negative.tif")
negroi = neg.getRoi()
pos.close()
IJ.run("Select None", "")

channels = ChannelSplitter.split(neg)

neg.close()
image = channels[3]
channels = channels[0:3]
proc = image.getProcessor()
directional_op = ImagePlus("directional_op", proc)

IJ.run(directional_op, "Directional Filtering", "type=Max operation=Opening line=30 direction=32")
directional_op = WindowManager.getImage("directional_op-directional")

tubes = range(5, 130, 12)

img_source = ImagePlus("image", proc)
src = clij2.push(img_source)
dst = clij2.create(src)
sigma = 2
clij2.gaussianBlur2D(src, dst, sigma, sigma)
img_blur2 = clij2.pull(dst)
src.close()
dst.close()

print("Tubeness mt start")
tubenesses = [None] * len(tubes)
rang = range(len(tubes))
threads = []
for i in rang:
    threads.append(threading.Thread(target=run_tube, args=(img_blur2, tubes[i], i, tubenesses)) )
    threads[i].start()

[x.join() for x in threads]
print("Tubeness all done")
print(tubenesses)

src = clij2.push(img_source)
dst = clij2.create(src)
sigmas = [5, 20]
imgsigmas = []
for sigma in sigmas:
    clij2.gaussianBlur2D(src, dst, sigma, sigma)
    img = clij2.pull(dst)
    imgsigmas.append(img)
print("Gaussian Blur done")
src.close()
dst.close()

variances = [5, 20]
imgvars = []
for variance in variances:
    img = ImagePlus("image", proc)
    IJ.run(img, "Variance...", "radius="+str(variance))
    imgvars.append(img)
print("Gaussian Blur done")

featuresArray = FeatureStackArray(image.getStackSize())
stack = ImageStack(image.getWidth(), image.getHeight())
# add new feature here (2/2) and do not forget to add it with a
# unique slice label!
stack.addSlice("directional_op", directional_op.getProcessor())
for i in range(len(sigmas)):
    stack.addSlice("sigma"+str(sigmas[i]), imgsigmas[i].getProcessor())

for i in range(len(tubes)):
    stack.addSlice("Tubeness"+str(tubes[i]), tubenesses[i].getProcessor())

for i in range(len(variances)):
    stack.addSlice("Variance"+str(variances[i]), imgvars[i].getProcessor())

for i in range(len(channels)):
    stack.addSlice("channel"+str(i+1), channels[i].getProcessor())

del sigmas
del tubes
del variances
del channels

# create empty feature stack
features = FeatureStack(stack.getWidth(), stack.getHeight(), False)

# set my features to the feature stack
features.setStack(stack)
# put my feature stack into the array
featuresArray.set(features, 0)
featuresArray.setEnabledFeatures(features.getEnabledFeatures())

mp = MultilayerPerceptron()
hidden_layers = "%i,%i,%i" % (20, 14, 8)
mp.setHiddenLayers(hidden_layers)
mp.setLearningRate(0.7)
mp.setDecay(True)
mp.setTrainingTime(200)
mp.setMomentum(0.3)

wekaSegmentation = WekaSegmentation(image)
wekaSegmentation.setFeatureStackArray(featuresArray)
wekaSegmentation.setClassifier(mp)

wekaSegmentation.addExample(0, posroi, 1)
wekaSegmentation.addExample(1, negroi, 1)
wekaSegmentation.trainClassifier()
wekaSegmentation.saveClassifier(folder + "\\vessel-classifier_big.model")
output = wekaSegmentation.applyClassifier(image, featuresArray, 0, True)

output.show()
