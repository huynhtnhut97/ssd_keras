import numpy as np
import cv2
import glob
import errno
import os
import math
import time

print ('-----------------------------------SCRIPT 01-----------------------------------')
print ('-------------CONVERT ANNOTATION FROM VISDRONE TO TURICREATE FORMAT-------------')
path = "/mnt/data/visdrone2018/gdown.pl/VisDrone2019-VID-val/annotations/*.txt"
path_image = "/mnt/data/visdrone2018/gdown.pl/VisDrone2019-VID-val/sequences/"

files = glob.glob(path)

def class_name(i):
	switcher={
		1:'pedestrian',
		2:'people',
		3:'bicycle',
		4:'car',
		5:'van',
		6:'truck',
		7:'tricycle',
		8:'awning-tricycle',
		9:'bus',
		10:'motor'
	}
	return switcher.get(i,"Invalid class")

def convert_str(object):
	converted_str = '<object>' + \
										 '<name>' + object['name'] + '</name>' + \
										 '<pose>' + object['pose'] + '</pose>' + \
										 '<truncated>' + object['truncated'] + '</truncated>' + \
										 '<difficult>' + object['difficult'] + '</difficult>' + \
										 '<bndbox>' + object['bndbox'] + '</bndbox>' + \
									'</object>'  
	return converted_str

def create_new(object):
	converted_str = '<annotation>' + \
											 '<folder>Visdrone2018</folder>' + \
											 '<filename>' + object['filename'] + '</filename>' + \
											 '<source>' + \
													 '<database>Visdrone2018 Database</database>' + \
													 '<image>flickr</image>' + \
											 '</source>' + \
											 '<size>' + \
													 '<width>' + object['width'] + '</width>' + \
													 '<height>' + object['height'] + '</height>' + \
													 '<depth>' + object['depth'] + '</depth>' + \
											 '</size>' + \
											 '<segmented>0</segmented>' + \
											 '<object>' + \
													 '<name>' + object['name'] + '</name>' + \
													 '<pose>' + object['pose'] + '</pose>' + \
													 '<truncated>' + object['truncated'] + '</truncated>' + \
													 '<difficult>' + object['difficult'] + '</difficult>' + \
													 '<bndbox>' + object['bndbox'] + '</bndbox>' + \
											 '</object>' + \
									'</annotation>'   
	return converted_str

for name in files:
		try:
				with open(name,'r') as f:
						pass # do what you want
						
						# Image name not includes extended
						# File frame image
						base = os.path.splitext(os.path.basename(name))[0]

						image = cv2.imread("/mnt/data/visdrone2018/gdown.pl/VisDrone2019-VID-train/sequences/uav0000360_00001_v/0000418.jpg")
						
						height = np.size(image, 0)
						width = np.size(image, 1)
						depth = np.size(image, 2) 
						
						list_frame=[]  
						imax=0
						
						# Read all lines in file
						#lines = f.read().split("\n")

						for line in f:
							(frame_id,target_id,bbox_left,bbox_top,bbox_width,bbox_height,score,object_category,truncation,occlusion ) = map(int,(line.split(',')))
							
							# Check if frame has person (a line in file)
							if (object_category != 0 and object_category != 11):
								# An object in a frame of video
								objects={
									#'image':'',
									'filename': base + '_' + str(frame_id),
									'width': str(width),
									'height': str(height),
									'depth': str(depth),
									'name': class_name(int(object_category)),
									'pose': 'Unspecified',
									'truncated': '0',
									'difficult': '0',
									'bndbox': ''
								}
								
							 # objects['image'] = 'Height: ' + str(height) + ' Width: ' + str(width)
							 # objects['name'] = class_name(frame_info[7])
								
								coordinates={
									'ymin': int(bbox_top),
									'xmin': int(bbox_left),
									'ymax': int(bbox_height) + int(bbox_top),
									'xmax': int(bbox_width) + int(bbox_left),
								}                
								
#                object_annotation = {
#                                      'label': 'person',
#                                      'coordinates': coordinates
#                                    }    

								coordinate_str = '<xmin>' + str(coordinates['xmin']) + '</xmin>' + \
																 '<ymin>' + str(coordinates['ymin']) + '</ymin>' + \
																 '<xmax>' + str(coordinates['xmax']) + '</xmax>' + \
																 '<ymax>' + str(coordinates['ymax']) + '</ymax>' 
								objects['bndbox'] = coordinate_str
								
								if (int(frame_id) <= len(list_frame) ):
									# Update
									list_frame[int(frame_id) - 1]['anno'] = list_frame[int(frame_id) - 1]['anno'].replace('</annotation>', '')
									list_frame[int(frame_id) - 1]['anno'] = list_frame[int(frame_id) - 1]['anno'] + convert_str(objects) + '</annotation>'
														
								else:
									print(objects['name'])
									# New element
									list_frame.append({
																		'name': objects['filename'],
																		'anno': create_new(objects)
																	 })
						converted_dir = '/mnt/data/nhuthuynh/val_converted_xml/' + base           
						if not os.path.exists(converted_dir):
							os.makedirs(converted_dir)
						

						for frame in list_frame:
#              if not os.path.exists(os.path.join(converted_dir,frame['name'], )):
#                os.makedirs(os.path.join(train_dir,class_name))
							
							fconvert = open(os.path.join(converted_dir,frame['name'] + '.xml'), 'a+')
							fconvert.write('%s\n' %str(frame['anno']))
							fconvert.close()    
						print('Done %s' %os.path.basename(name))
		except IOError as exc:
				if exc.errno != errno.EISDIR:
						raise