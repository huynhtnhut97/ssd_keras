import time
import json
import argparse
import os
from operator import itemgetter
import shutil
def createImages(path,dest):
	#Copy images from sequences to JPEGImages
	for folder in os.listdir(path):
		folderPath = os.path.join(path,folder)
		if (os.path.isdir(folderPath)):
			for filename in os.listdir(folderPath):
				filePath = os.path.join(folderPath,filename)
				formatFilename = int(os.path.splitext(filename)[0])
				newFilename = folder+'_'+str(formatFilename)+'.jpg'
				newFilePath = os.path.join(dest,newFilename)
				if(os.path.isfile(filePath)):
					shutil.copy(filePath,newFilePath)
def createAnnotation(path,dest):
	#Copy xml annotation to Annotations
	for folder in os.listdir(path):
		folderPath = os.path.join(path,folder)
		if (os.path.isdir(folderPath)):
			for filename in os.listdir(folderPath):
				filePath = os.path.join(folderPath,filename)
				newFilePath = os.path.join(dest,filename)
				if(os.path.isfile(filePath)):
					shutil.copy(filePath,newFilePath)
def createTrainAndTestFile(source,dest,role):
	#Create val and train file for Main
	classes = ["ignore_region","pedestrian","people","bicycle","car","van","truck","tricycle","awning-tricycle","bus","motor","others"]
	ground_truth = []
	object_Category = 0
	path = source
	for filename in os.listdir(path):
		#frameID = 0
		filePath = os.path.join(path,filename)
		if (os.path.isfile(filePath)):
			with open(os.path.join(path,filename), 'r') as f:
				for line in f:
					(frame_id,target_id,bbox_left,bbox_top,bbox_width,bbox_height,score,object_category,truncation,occlusion ) = map(int,(line.split(',')))
					#if(object_category!=0 and object_category !=11):
					if(object_category!=0 and object_category!=11):
						fname = os.path.splitext(filename)[0]+'_'+str(frame_id)
						ground_truth.append(list((fname,frame_id,target_id,bbox_left,bbox_top,bbox_width,bbox_height,score,object_category,truncation,occlusion)))
	
	sortedList = sorted(ground_truth, key=itemgetter(8))
	for obj in sortedList:

		# FrameID = obj[0]
		# bbox_left = obj[2]
		# bbox_top = obj[3]
		# bbox_width = obj[4]
		# bbox_height = obj[5]
		fname = obj[0]
		current_Object_category = obj[8]
		print(current_Object_category)
		#pathtoDir = os.path.join(path,os.path.splitext(filename)[0])
		if(current_Object_category>object_Category):
			object_Category = current_Object_category
			print(object_Category)
			class_name = classes[object_Category]
			txtFile = open(os.path.join(dest,class_name+'_{}.txt'.format(role)),'w')
		txtFile.write(str(fname)+' '+str(1))
		txtFile.write('\n')
	#print("Total frames in video {} is {}".format(os.path.splitext(filename)[0],frameID))
def arg_parse():
	"""
	Parse arguements to the detect module
	
	"""
	parser = argparse.ArgumentParser(description='YOLO v3 Video Detection Module')
   
	parser.add_argument("--imgs", dest = 'imgsource',
						default = "./image", type = str)
	parser.add_argument("--imgsDest", dest = 'imgdest',
						default = "./image", type = str)

	parser.add_argument("--annos", dest = 'annosource',
						default = "./annotations", type = str)
	parser.add_argument("--annosDest", dest = 'annodest',
						default = "./annotations", type = str)

	parser.add_argument("--trainTestDest", dest = 'traintestdest',
						default = "./image", type = str)
	parser.add_argument("--originAnno", dest = 'originanno',
						default = "./image", type = str)
	return parser.parse_args()
if __name__=="__main__":
	args = arg_parse()
	pathToImgSource = args.imgsource
	pathToImgDest = args.imgdest
	pathToAnnoSource = args.annosource
	pathToAnnoDest = args.annodest
	trainTestDest = args.traintestdest
	originAnno = args.originanno
	#createImages(pathToImgSource,pathToImgDest)
	#createAnnotation(pathToAnnoSource,pathToAnnoDest)
	#createTrainAndTestFile(originAnno,trainTestDest,'train')
	createTrainAndTestFile(originAnno,trainTestDest,'val ')