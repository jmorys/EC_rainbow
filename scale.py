from ij import WindowManager
from ij import IJ
from net.imglib2.img import ImagePlusAdapter
from net.imglib2.img.display.imagej import ImageJFunctions
from fiji.process import Image_Expression_Parser
from ij.io import DirectoryChooser, OpenDialog
from ij.plugin import ChannelSplitter


od = OpenDialog("Choose a file", None)
file = od.getFileName()
dc = DirectoryChooser("Choose a folder")
folder = dc.getDirectory()
imp = IJ.openImage(folder+file)
channels = ChannelSplitter.split(imp)

IJ.save(channels[3], folder + "\\vessels.tif")


listt = channels[:3]
listt_2 = []
for i in range(0, 3):
    listt_2.append(ImagePlusAdapter.wrap(listt[i]))
map = {'A': listt_2[0], 'B': listt_2[1], 'C': listt_2[2]}
expression = '(A+B+C)/3'
# Instantiate plugin
parser = Image_Expression_Parser()
# Configure & execute
parser.setImageMap(map)
parser.setExpression(expression)
parser.process()
result = parser.getResult()
result_imp = ImageJFunctions.show(result)
IJ.run('Rename...', 'title=AverageIMG')
IJ.run("16-bit")
average_imp = WindowManager.getImage("AverageIMG")
IJ.save(average_imp, folder + "\\AverageIMG.tif")
result_img = ImagePlusAdapter.wrap(result_imp)
mean_average = average_imp.getStatistics()
mean_average = mean_average.mean
scaled_list = []
for i in range(0, 3):
    imp = listt[i]
    img = listt_2[i]
    map = {'A': img, 'B': result_img}
    mean_img = imp.getStatistics()
    mean_img = mean_img.mean
    expression = 'A-(0.9*' + str(mean_img) + "/" + str(mean_average) + "*B)"
    parser = Image_Expression_Parser()
    # Configure & execute
    parser.setImageMap(map)
    parser.setExpression(expression)
    parser.process()
    result = parser.getResult()  # is an ImgLib image
    result_imp = ImageJFunctions.show(result)
    title = 'title=C'+str(i+1)+'wo_scaled_bg'
    IJ.run('Rename...', title)
    IJ.run("Min...", "value=0")
    IJ.run("16-bit")
average_imp.close()
IJ.run("Merge Channels...", "c1=C1wo_scaled_bg c2=C2wo_scaled_bg c3=C3wo_scaled_bg create")
composite = WindowManager.getImage("Composite")
IJ.save(composite, folder + "\\C123_wo_scaled_bg.tif")
composite.close()
