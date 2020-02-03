import os
from os import listdir
from os.path import isfile, join
import fnmatch
import xml.etree.ElementTree as ET
path = "/mnt/data/nhuthuynh/visdrone-data-VOC-format/VOCdevkit/VOC2007/JPEGImages/"
txtFile = open('train.txt','w')
for file in listdir(path):
		if '.jpg' == os.path.splitext(file)[1]:
			annofile = os.path.join("/mnt/data/nhuthuynh/visdrone-data-VOC-format/VOCdevkit/VOC2007/Annotations/",os.path.splitext(file)[0]+".xml")
			if os.path.isfile(annofile):
				try:
					ET.parse(annofile)
					txtFile.write(os.path.splitext(file)[0]+'\n')
				except:
					print("Cant parse file: {}".format(os.path.basename(annofile)))
					continue
# file = listdir(path)[-1]
# txtFile.write(os.path.join(path,os.path.splitext(file)[0])+'\n')
txtFile.close()
