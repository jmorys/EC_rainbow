from ij import ImagePlus, IJ, WindowManager, ImageStack
from ij.process import FloatProcessor, ShortProcessor
from ij.io import DirectoryChooser
from ij.plugin import ChannelSplitter, ImageCalculator
from trainableSegmentation import WekaSegmentation, FeatureStackArray, FeatureStack
from java.lang import System
# vessel detection training
from inra.ijpb import morphology
# MorpholibJ package must be activated


def VesselFinder(channel_array, classifier_path):
    channels = channel_array
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
        IJ.run(img, "Gaussian Blur...", "sigma="+str(sigma))
        imgsigmas.append(img)
    print("Gaussian Blur done")

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
    del stack

    wekaSegmentation = WekaSegmentation(image)
    wekaSegmentation.setFeatureStackArray(featuresArray)
    wekaSegmentation.loadClassifier(classifier_path + "\\vessel-classifier_big.model")
    output = wekaSegmentation.applyClassifier(image, featuresArray, 0, True)
    System.gc()
    return output


dc = DirectoryChooser("Choose a classifier folder")
classifier_path = dc.getDirectory()


dc = DirectoryChooser("Choose a folder")
folder = dc.getDirectory()
comp = IJ.openImage(folder+"\\C123_wo_scaled_bg.tif")
channels_raw = ChannelSplitter.split(comp)
del comp
imp = IJ.openImage(folder+"\\vessels.tif")
channels = []
for i in range(len(channels_raw)):
    channels.append(channels_raw[i])
print(len(channels))
del channels_raw
channels.append(imp)

dimensions = imp.getDimensions()
imp.close()
d1 = dimensions[0]
lim1 = dimensions[1] // 4
lim2 = lim1*2
lim3 = lim1*3
s = 300

ranges = [[0, ((lim1+s)*d1)], [(lim1-s)*d1, ((lim2+s)*d1)],
          [(lim2-s)*d1, (lim3+s)*d1], [(lim3-s) * d1, dimensions[1] * d1]]
image_frags = []
for i in range(len(channels)):
    channel_frags = []
    channel = channels[i]
    pix = channel.getProcessor().getPixels()
    cm = channel.getProcessor().getColorModel()
    for r in range(len(ranges)):
        frag_pix = pix[ranges[r][0]:ranges[r][1]]
        channel_frags.append(ImagePlus("Random", ShortProcessor(d1, (ranges[r][1]-ranges[r][0])/d1, frag_pix, cm)))
    image_frags.append(channel_frags)
out_frags = []
for r in range(len(ranges)):
    channel_array = [image_frags[0][r], image_frags[1][r], image_frags[2][r], image_frags[3][r]]
    imp = VesselFinder(channel_array, classifier_path)
    imp = imp.getStack()
    imp = imp.getProcessor(1)
    cm = imp.getColorModel()
    out_frags.append(imp)


cm = out_frags[0].getColorModel()
mranges = [[0, (-s*d1)], [s * d1, (-s * d1)], [s * d1, (-s * d1)], [s * d1, (dimensions[1] * d1 - (lim3 - s) * d1)]]
merged_pix = []
for r in range(len(ranges)):
    merged_pix.extend(out_frags[r].getPixels()[mranges[r][0]:mranges[r][1]])

imp = ImagePlus("Result", FloatProcessor(d1, dimensions[1], merged_pix, cm))
IJ.save(imp, folder + "\\vessels_RAW.tif")
IJ.run(imp, "Median...", "radius=2")
strel = morphology.Strel.Shape.DISK.fromDiameter(5)
result = morphology.Morphology.closing(imp.getProcessor(), strel)
IJ.setAutoThreshold(imp, "Default dark")
IJ.run(imp, "Convert to Mask", "")
strel = morphology.Strel.Shape.DISK.fromDiameter(20)
result = morphology.Morphology.closing(result, strel)
imp.close()
strel = morphology.Strel.Shape.DISK.fromDiameter(3)
result = morphology.Morphology.dilation(result, strel)
imp = ImagePlus("vessels", result)
IJ.run(imp, "Median...", "radius=20")
imp.show()


ImageCalculator = ImageCalculator()
IJ.run(imp, "Analyze Particles...", "size=0-2000 show=Masks display")

imp1 = WindowManager.getImage("Mask of vessels")
imp = ImageCalculator.run("Subtract", imp, imp1)
imp1.close()

IJ.run(imp, "Invert", "")
imp = WindowManager.getImage("vessels")
IJ.run(imp, "Analyze Particles...", "size=0-2000 show=Masks display")
imp1 = WindowManager.getImage("Mask of vessels")
imp = ImageCalculator.run("Subtract", imp, imp1)
imp1.close()
imp = WindowManager.getImage("vessels")
IJ.run(imp, "Analyze Particles...", "size=0-9000 circularity=0.00-0.60 show=Masks display")
imp1 = WindowManager.getImage("Mask of vessels")
imp = ImageCalculator.run("Subtract", imp, imp1)
type(imp)
imp1.close()
imp = WindowManager.getImage("vessels")
IJ.run(imp, "Invert", "")
IJ.save(imp, folder + "\\vessels_result.tif")
