from ij import IJ, WindowManager
from ij.measure import ResultsTable
from ij.plugin.frame import RoiManager
from ij.io import DirectoryChooser
from ij.plugin import ImageCalculator
import csv
import time
import os

rm = RoiManager.getInstance()
if rm is None:
    RoiManager()
    rm = RoiManager.getInstance()
rm.reset()
rstable = ResultsTable()

ImageCalculator = ImageCalculator()
dc = DirectoryChooser("Choose a folder")
folder = dc.getDirectory()
mask = IJ.openImage(folder + "vessels_result.tif")
mask.setTitle("Mask")
for i in range(1, 4):
    imp = IJ.openImage(folder + "\\split\\C" + str(i) + "\\result_READY.tif")
    imp.setTitle("C")
    imp.show()
    imp = ImageCalculator.run("Min", imp, mask)
    imp = WindowManager.getImage("C")
    IJ.run(imp, "Find Maxima...", "prominence=20 output=[Point Selection]")
    mroi = imp.getRoi()

    IJ.run("Measure")
    table = rstable.getResultsTable()
    X = table.getColumn(6)
    Y = table.getColumn(7)

    with open(folder + "\\split\\C" + str(i) + "\\cells.csv", 'wb') as csvfile:
        w = csv.writer(csvfile, delimiter=',', quotechar="\"", quoting=csv.QUOTE_NONNUMERIC)
        w.writerow(X)
        w.writerow(Y)
    rm.addRoi(mroi)
    imp.close()
    IJ.run("Clear Results")
mask.show()
rm.setSelectedIndexes(range(3))
rm.runCommand("Combine")

# wszystkie komórki
IJ.run("Measure")
table = rstable.getResultsTable()
X = table.getColumn(6)
Y = table.getColumn(7)

with open(folder + "\\cell_positions.csv", 'wb') as csvfile:
    w = csv.writer(csvfile, delimiter=',', quotechar="\"", quoting=csv.QUOTE_NONNUMERIC)
    w.writerow(X)
    w.writerow(Y)

mroi = mask.getRoi()
mask.deleteRoi()
rm.reset()
rm.addRoi(mroi)
IJ.run("Clear Results")
dist_folder = folder+"\\distances"


# zapisywanie odległości w obrębie segmentu
if not os.path.exists(dist_folder):
    os.mkdir(dist_folder)

IJ.run(mask, "Analyze Particles...", "display add")
IJ.run("Clear Results")
dimentions = mask.getDimensions()
marker = IJ.createImage("marker", dimentions[0], dimentions[1], 1, 8)
marker.show()
rm = RoiManager.getInstance()
length = rm.getCount()

s = "marker=marker_c mask=mask_c distances=[Chessknight (5,7,11)] output=[16 bits]"
start_time = time.time()
for i in range(1, length):
    rm.setSelectedIndexes([0, i])
    rm.runCommand("AND")
    IJ.run("Measure")
    table = rstable.getResultsTable()
    X = table.getColumn(6)
    if X is None:
        IJ.run("Clear Results")
        continue
    if len(X) == 1:
        IJ.run("Clear Results")
        continue
    Y = table.getColumn(7)
    rm.deselect()
    print(X)
    data = [X, Y]
    IJ.run("Clear Results")

    mroi = rm.getRoi(i)
    mask.setRoi(mroi)
    IJ.run(mask, "Duplicate...", "title=mask_c")
    mask_c = WindowManager.getImage("mask_c")

    marker.setRoi(mroi)
    IJ.run(marker, "Duplicate...", "title=marker_c")
    marker_c = WindowManager.getImage("marker_c")

    coord = mroi.getBounds()

    for j in range(len(X)):
        marker_c.getProcessor().putPixel(int(X[j] - coord.x), int(Y[j] - coord.y), 255)
        marker_c.updateAndDraw()

        IJ.run("Geodesic Distance Map", s)
        marker_c.getProcessor().putPixel(int(X[j] - coord.x), int(Y[j] - coord.y), 0)
        geod = WindowManager.getImage("mask_c-geoddist")
        points = []
        for k in range(len(X)):
            points.append(geod.getProcessor().getPixel(int(X[k] - coord.x), int(Y[k] - coord.y)))
        geod.close()
        data.append(points)
    marker_c.close()
    mask_c.close()
    with open(dist_folder + "\\dist_" + str(i) + ".csv", 'wb') as csvfile:
        w = csv.writer(csvfile, delimiter=',', quotechar="\"", quoting=csv.QUOTE_NONNUMERIC)
        for k in range(len(data)):
            w.writerow(data[k])

marker.close()

end_time = time.time()
print("lasted" + str(end_time-start_time))
