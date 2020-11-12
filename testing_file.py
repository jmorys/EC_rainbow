from net.imagej.ops import Ops
from ij import ImagePlus, IJ, WindowManager, ImageStack
from ij.io import DirectoryChooser
from ij.plugin import ChannelSplitter
from trainableSegmentation import WekaSegmentation, FeatureStackArray, FeatureStack
from weka.classifiers.functions import SMO
from ij.plugin.filter import Convolver
import multiprocessing


# dc = DirectoryChooser("Choose a folder")
# folder = dc.getDirectory()
# pos = IJ.openImage(folder + "\\train_cells.tif")
# posroi = pos.getRoi()
# neg = IJ.openImage(folder + "\\train_bg.tif")
# negroi = neg.getRoi()
# pos.close()
# neg.close()
#
# train = IJ.openImage(folder + "\\train_data_base.tif")
# train = ChannelSplitter.split(train)
# train = train[0]







