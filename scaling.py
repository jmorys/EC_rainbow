from ij import WindowManager
from ij import IJ
from net.imglib2.img import ImagePlusAdapter
from net.imglib2.img.display.imagej import ImageJFunctions
from fiji.process import Image_Expression_Parser
from spimopener import Rename
from ij import measure
listt = []
listt_2 = []
for i in range(0,3):
	listt.append(WindowManager.getImage(i+1))
	listt_2.append(ImagePlusAdapter.wrap(listt[i]))
map = {'A': listt_2[0], 'B': listt_2[1], 'C':listt_2[2]}
expression = '(A+B+C)/3'
# Instantiate plugin
parser = Image_Expression_Parser()
# Configure & execute
parser.setImageMap(map)
parser.setExpression(expression)
parser.process()
result = parser.getResult() # is an ImgLib image

result_imp= ImageJFunctions.show(result)
IJ.run ('Rename...', 'title=AverageIMG')
result_img= ImagePlusAdapter.wrap(result_imp)
mean_average= result_imp.getStatistics()
mean_average= mean_average.mean
scaled_list=[]
for i in range(0,3):
    imp = listt[i]
    img = listt_2[i]
    map = {'A': img, 'B': result_imp}
    mean_img = imp.getStatistics()
    mean_img=mean_img.mean
    expression = 'A-(0.9*mean_img/mean_average*B)'
    parser = Image_Expression_Parser()
    # Configure & execute
    parser.setImageMap(map)
    parser.setExpression(expression)
    parser.process()
    result = parser.getResult() # is an ImgLib image
    title='title=C'+str(i+1)+'wo_scaled_bg'
    IJ.run ('Rename...', title)
