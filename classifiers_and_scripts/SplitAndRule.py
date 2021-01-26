from ij import ImagePlus, IJ
from ij.process import FloatProcessor, ShortProcessor
from ij.io import DirectoryChooser
import os
from ij.plugin import ChannelSplitter
from trainableSegmentation import WekaSegmentation
from inra.ijpb import morphology
from java.lang import System

dc = DirectoryChooser("Choose a folder")
folder = dc.getDirectory()

sub = folder+"\\split"

if not os.path.exists(sub):
    os.mkdir(sub)

scaled = "C123_wo_scaled_bg.tif"
imp = IJ.openImage(os.path.join(folder+scaled))
channels = ChannelSplitter.split(imp)

dimentions = imp.getDimensions()
imp.close()
d1 = dimentions[0]
lim1 = dimentions[1]//4
lim2 = lim1*2
lim3 = lim1*3
s = 300

pathh = "C:\\Users\\USER\\Documents\\studia\\zaklad\\EC_rainbow\\"
segmentator = WekaSegmentation()
segmentator.loadClassifier(pathh)


for i in range(len(channels)):
    channel = channels[i]
    pix = channel.getProcessor().getPixels()
    cm = channel.getProcessor().getColorModel()
    pix1 = pix[:((lim1+s)*d1)]
    pix2 = pix[((lim1-s)*d1):((lim2+s)*d1)]
    pix3 = pix[((lim2-s)*d1):((lim3+s)*d1)]
    pix4 = pix[((lim3-s)*d1):]
    imps = [ImagePlus("Random", ShortProcessor(d1, (lim1 + s), pix1, cm)),
            ImagePlus("Random", ShortProcessor(d1, (lim2 - lim1 + 2 * s), pix2, cm)),
            ImagePlus("Random", ShortProcessor(d1, (lim3 - lim2 + 2 * s), pix3, cm)),
            ImagePlus("Random", ShortProcessor(d1, (dimentions[1] - lim3 + s), pix4, cm))]
    subsub = sub+"\\C"+str(i+1)
    del pix1
    del pix2
    del pix3
    del pix4
    if not os.path.exists(subsub):
        os.mkdir(subsub)
    pixels = []
    for j in range(4):
        System.gc()
        imp = segmentator.applyClassifier(imps[j], 0, True)
        imp = imp.getStack()
        imp = imp.getProcessor(1)
        cm = imp.getColorModel()
        pixels.append(imp.getPixels())
    pix1 = pixels[0]
    pix2 = pixels[1]
    pix3 = pixels[2]
    pix4 = pixels[3]
    pixcomb = pix1[:-s*d1]+pix2[s*d1:-s*d1]+pix3[s*d1:-s*d1]+pix4[s*d1:]
    result = ImagePlus("Result", FloatProcessor(d1, dimentions[1], pixcomb, cm))
    IJ.save(result, subsub + "\\result_RAW.tif")
    IJ.run(result, "Despeckle", "")
    strel = morphology.Strel.Shape.DISK.fromDiameter(3)
    result = morphology.Morphology.erosion(result.getProcessor(), strel)
    result = result.convertToByteProcessor()
    result = ImagePlus("plus", result)
    IJ.run(result, "Gaussian Blur...", "sigma=15")
    IJ.save(result, subsub + "\\result_READY.tif")
    result.close()
