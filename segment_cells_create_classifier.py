from ij import ImagePlus, IJ, WindowManager, ImageStack
from ij.io import DirectoryChooser
from ij.plugin import ChannelSplitter
from trainableSegmentation import WekaSegmentation, FeatureStackArray, FeatureStack
from weka.classifiers.functions import MultilayerPerceptron

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
directional_op.hide()

tubes = range(5, 130, 12)
tubenesses = []
for tube in tubes:
    img = ImagePlus("image", proc)
    IJ.run(img, "Gaussian Blur...", "sigma=2")
    IJ.run(img, "Tubeness", "sigma=" + str(tube) + " use")
    img = WindowManager.getImage("tubeness of image")
    img.hide()
    img = img.getProcessor().convertToShortProcessor()
    img = ImagePlus("img", img)
    tubenesses.append(img)
    IJ.run(img, "Gaussian Blur...", "sigma=2")
    print("Tubeness" + str(tube) + "done")

sigmas = [5, 20]
imgsigmas = []
for sigma in sigmas:
    img = ImagePlus("image", proc)
    IJ.run(img, "Gaussian Blur...", "sigma=" + str(sigma))
    imgsigmas.append(img)
print("Gaussian Blur done")

variances = [5, 20]
imgvars = []
for variance in variances:
    img = ImagePlus("image", proc)
    IJ.run(img, "Variance...", "radius=" + str(variance))
    imgvars.append(img)
print("Gaussian Blur done")

featuresArray = FeatureStackArray(image.getStackSize())
stack = ImageStack(image.getWidth(), image.getHeight())
# add new feature here (2/2) and do not forget to add it with a
# unique slice label!
stack.addSlice("directional_op", directional_op.getProcessor())
for i in range(len(sigmas)):
    stack.addSlice("sigma" + str(sigmas[i]), imgsigmas[i].getProcessor())

for i in range(len(tubes)):
    stack.addSlice("Tubeness" + str(tubes[i]), tubenesses[i].getProcessor())

for i in range(len(variances)):
    stack.addSlice("Variance" + str(variances[i]), imgvars[i].getProcessor())

for i in range(len(channels)):
    stack.addSlice("channel" + str(i + 1), channels[i].getProcessor())

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
