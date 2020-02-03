import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TerminateOnNaN, CSVLogger
from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
from math import ceil
import numpy as np
from matplotlib import pyplot as plt

from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization

from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.data_augmentation_chain_original_ssd import SSDDataAugmentation
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms


# Set a few configuration parameters.
img_height = 300
img_width = 300
n_classes = 10
model_mode = 'training'
normalize_coords = True

# TODO: Set the path to the `.h5` file of the model to be loaded.
model_path = "/mnt/data/nhuthuynh/SSD/models/ssd300_visdrone2019_epoch-70_loss-4.8976_val_loss-8.2627.h5"

# We need to create an SSDLoss object in order to pass that to the model loader.
ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

K.clear_session() # Clear previous models from memory.

model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
                                               'L2Normalization': L2Normalization,
                                               'DecodeDetections': DecodeDetections,
                                               'compute_loss': ssd_loss.compute_loss})

images_path = "/mnt/data/nhuthuynh/sequences/"
results_path = "./results"
list_file = []
for folder in os.listdir(images_path):
    folder_path = os.path.join(images_path,folder)
    for image in os.listdir(folder_path):
        image_path = os.path.join(folder_path,image)
        print(image_path)
        list_file.append(image_path)
val_dataset = DataGenerator(load_images_into_memory=True, filenames=list_file, hdf5_dataset_path=None)

convert_to_3_channels = ConvertTo3Channels()
resize = Resize(height=img_height, width=img_width)
predict_generator = val_dataset.generate(batch_size=1,
                                         shuffle=True,
                                         transformations=[convert_to_3_channels,
                                                          resize],
                                         label_encoder=None,
                                         returns={'processed_images',
                                                  'filenames',
                                                  'inverse_transform',
                                                  'original_images'},
                                         keep_images_without_gt=True)
batch_images, batch_filenames, batch_inverse_transforms, batch_original_images = next(predict_generator)
classes = ["ignore_region","pedestrian","people","bicycle","car","van","truck","tricycle","awning-tricycle","bus","motor","others"]

for i in range(len(list_file)):
  print(i)
  print("Image:", batch_filenames[i])
  y_pred = model.predict(batch_images)
  y_pred_decoded = decode_detections(y_pred,
                                     confidence_thresh=0.5,
                                     iou_threshold=0.4,
                                     top_k=200,
                                     normalize_coords=normalize_coords,
                                     img_height=img_height,
                                     img_width=img_width)
  y_pred_decoded_inv = apply_inverse_transforms(y_pred_decoded, batch_inverse_transforms)
  img_name = os.path.basename(batch_filenames[i])
  video_name = os.path.basename(os.path.dirname(batch_filenames[i]))
  file_name = video_name+'_'+os.path.splitext(img_name)[0]+'.txt' #detected result file
  file_path = os.path.join('./results',file_name)
  with open(file_path,'w') as file:
    for box in y_pred_decoded_inv[i]:
        xmin = box[2]
        ymin = box[3]
        xmax = box[4]
        ymax = box[5]
        w = abs(xmax-xmin)
        h = abs(ymax-ymin)
        label = '{}'.format(classes[int(box[0])])
        file.write(str(labels)+','+str(int(x))+','+str(int(y))+','+str(int(w))+','+str(int(h)))
        file.write('\n')