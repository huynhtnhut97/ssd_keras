import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from keras import backend as K
from keras.models import load_model
from keras.optimizers import Adam
from scipy.misc import imread
import numpy as np
from matplotlib import pyplot as plt

from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization
from data_generator.object_detection_2d_data_generator import DataGenerator
from eval_utils.average_precision_evaluator import Evaluator

# Set a few configuration parameters.
img_height = 300
img_width = 300
n_classes = 10
model_mode = 'training'

'''LOAD TRAINED SSD'''
 #Build the model and load trained weights into it
 # 1: Build the Keras model

K.clear_session() # Clear previous models from memory.

model = ssd_300(image_size=(img_height, img_width, 3),
                n_classes=n_classes,
                mode=model_mode,
                l2_regularization=0.0005,
                scales=[0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05], # The scales for MS COCO [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
                aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5]],
                two_boxes_for_ar1=True,
                steps=[8, 16, 32, 64, 100, 300],
                offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                clip_boxes=False,
                variances=[0.1, 0.1, 0.2, 0.2],
                normalize_coords=True,
                subtract_mean=[123, 117, 104],
                swap_channels=[2, 1, 0],
                confidence_thresh=0.01,
                iou_threshold=0.45,
                top_k=200,
                nms_max_output_size=400)

# 2: Load the trained weights into the model.

# TODO: Set the path of the trained weights.
weights_path = "/mnt/data/nhuthuynh/SSD/models/ssd300_visdrone2019_epoch-70_loss-4.8976_val_loss-8.2627.h5"
model.load_weights(weights_path, by_name=True)

# 3: Compile the model so that Keras won't complain the next time you load it.

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

'''Create a data generator for the evaluation dataset'''
dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path="/mnt/data/nhuthuynh/SSD/hdf5_dataset/visdrone2019_val.h5")
# TODO: Set the paths to the dataset here.
Pascal_VOC_dataset_images_dir = "/mnt/data/nhuthuynh/visdrone-data-VOC-format/VOCdevkit/VOC2007/JPEGImages"
Pascal_VOC_dataset_annotations_dir = "/mnt/data/nhuthuynh/visdrone-data-VOC-format/VOCdevkit/VOC2007/Annotations"
Pascal_VOC_dataset_image_set_filename = "/mnt/data/nhuthuynh/visdrone-data-VOC-format/VOCdevkit/VOC2007/ImageSets/Main/val.txt"

# # The XML parser needs to now what object class names to look for and in which order to map them to integers.
# classes = ["ignore_region","pedestrian","people","bicycle","car","van","truck","tricycle","awning-tricycle","bus","motor","others"]

# dataset.parse_xml(images_dirs=[Pascal_VOC_dataset_images_dir],
#                   image_set_filenames=[Pascal_VOC_dataset_image_set_filename],
#                   annotations_dirs=[Pascal_VOC_dataset_annotations_dir],
#                   classes=classes,
#                   include_classes='all',
#                   exclude_truncated=False,
#                   exclude_difficult=False,
#                   ret=False)

'''RUN EVALUATTION'''
evaluator = Evaluator(model=model,
                      n_classes=n_classes,
                      data_generator=dataset,
                      model_mode=model_mode)

results = evaluator(img_height=img_height,
                    img_width=img_width,
                    batch_size=32,
                    data_generator_mode='resize',
                    round_confidences=False,
                    matching_iou_threshold=0.5,
                    border_pixels='include',
                    sorting_algorithm='quicksort',
                    average_precision_mode='sample',
                    num_recall_points=11,
                    ignore_neutral_boxes=True,
                    return_precisions=True,
                    return_recalls=True,
                    return_average_precisions=True,
                    verbose=True)

mean_average_precision, average_precisions, precisions, recalls = results
classes = ["ignore_region","pedestrian","people","bicycle","car","van","truck","tricycle","awning-tricycle","bus","motor","others"]
'''VISUALIZE THE RESULT'''
for i in range(1, len(average_precisions)):
    print("{:<14}{:<6}{}".format(classes[i], 'AP', round(average_precisions[i], 3)))
print()
print("{:<14}{:<6}{}".format('','mAP', round(mean_average_precision, 3)))
