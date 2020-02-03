import sys
import os
import time
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

val_dataset = DataGenerator(load_images_into_memory=True, filenames=["/mnt/data/visdrone2018/gdown.pl/VisDrone2019-VID-val/sequences/uav0000305_00000_v/0000093.jpg"
], hdf5_dataset_path=None)

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
i = 0 # Which batch item to look at

print("Image:", batch_filenames[i])
start = time.time()
y_pred = model.predict(batch_images)
print("Processing time: {}".format(time.time()-start))

start = time.time()
y_pred = model.predict(batch_images)
print("Processing time: {}".format(time.time()-start))
# img = image.load_img("/mnt/data/visdrone2018/gdown.pl/VisDrone2019-VID-val/sequences/uav0000268_05773_v/0000571.jpg", target_size=(img_width, img_height))
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)

#boxes, scores, labels = model.predict(x)
# for box, score, label in zip(boxes[0], scores[0], labels[0]):
# 	print("Label: {}, score: {}".format(label,score))

# 4: Decode the raw predictions in `y_pred`.

y_pred_decoded = decode_detections(y_pred,
                                   confidence_thresh=0.5,
                                   iou_threshold=0.4,
                                   top_k=200,
                                   normalize_coords=normalize_coords,
                                   img_height=img_height,
                                   img_width=img_width)
y_pred_decoded_inv = apply_inverse_transforms(y_pred_decoded, batch_inverse_transforms)

np.set_printoptions(precision=2, suppress=True, linewidth=90)
print("Predicted boxes:\n")
print('   class   conf xmin   ymin   xmax   ymax')
print(y_pred_decoded_inv[i])

colors = plt.cm.hsv(np.linspace(0, 1, n_classes+1)).tolist()
classes = ["ignore_region","pedestrian","people","bicycle","car","van","truck","tricycle","awning-tricycle","bus","motor","others"]

plt.figure(figsize=(20,12))
plt.imshow(batch_original_images[i])
current_axis = plt.gca()
for box in y_pred_decoded_inv[i]:
    xmin = box[2]
    ymin = box[3]
    xmax = box[4]
    ymax = box[5]
    color = colors[int(box[0])]
    label = '{}'.format(classes[int(box[0])])
    current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))  
    current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'edgecolor':color, 'alpha':1.0})
plt.savefig("demo.png")