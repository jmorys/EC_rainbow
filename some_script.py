from ij import IJ, WindowManager, ImagePlus
from ij.measure import ResultsTable
from ij.plugin.frame import RoiManager
from ij.process import ByteProcessor
from ij.io import DirectoryChooser
from ij.plugin import ImageCalculator
import csv, os, time, random
from inra.ijpb import binary
from threading import Thread
# geodesicDistanceMap(ij.process.ImageProcessor marker, ij.process.ImageProcessor mask)


mask = WindowManager.getImage("mask")
mask_p = mask.getProcessor()
X=[4933,5489,7195,4944,3515,4201,7440,4885,5788,5738,4096,4243,7182,2262,2444,5443]
Y=[1871,2528,2620,2124,3943,2245,2044,1763,2351,3222,2229,10510,2734,6747,6014,2129]
pos = [X,Y]
dimentions = mask.getDimensions()

def thread_geod(mask_p, index, pos, results):
	shortWeights = binary.ChamferWeights.CHESSKNIGHT.getShortWeights()
	gt = binary.geodesic.GeodesicDistanceTransformShort5x5(shortWeights, False)
	marker = ByteProcessor(dimentions[0], dimentions[1])
	marker.putPixel(pos[0][index], pos[1][index], 255)
	proc=gt.geodesicDistanceMap(marker, mask_p)
	points=[]
	for w in range(len(X)):
		points.append(proc.getPixel(pos[0][w], pos[1][w]))
	results[index]= points

threads = [None] * len(X)
results = [None] * len(X)

for i in range(len(threads)):
	threads[i] = Thread(target= thread_geod, args=(mask_p, i, pos, results))
	threads[i].start()

for i in range(len(threads)):
    threads[i].join()

for i in range(len(results)):
	print(results[i])
