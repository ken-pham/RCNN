import os
import numpy as np
import tensorflow as tf
from Config import Config
import models
import data_generator
from dataset import Data
from utils import extract_bboxes
import cv2
import random
import colorsys
import matplotlib.pyplot as plt
from utils import compute_ap

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


def display_box_instances(image, boxes, masks, class_ids, class_name, scores = None):
    
    n_instances = boxes.shape[0]
    colors = random_colors(n_instances)

    if not n_instances:
        print('NO INSTANCES TO DISPLAY')
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    for i, color in enumerate(colors):
        if not np.any(boxes[i]):
            continue

        y1, x1, y2, x2 = boxes[i]
        label = class_name[class_ids[i]]
        score = scores[i] if scores is not None else None
        caption = '{} {:.2f}'.format(label, score) if score else label
        mask = masks[:, :, i]

        image = apply_mask(image, mask, color)
        
        color_rec = [int(c) for c in np.array(colors[i]) * 255]
        image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color_rec, 2)
        image = cv2.putText(
            image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, color = (255, 255, 255),lineType= cv2.LINE_AA)

    return image

class instance_custom_training:
    def __init__(self):
        self.model_dir = os.getcwd()

        
    def modelConfig(self,network_backbone = "resnet101",  num_classes =  1,  class_names = ["BG"], batch_size = 1, detection_threshold = 0.7, image_max_dim = 640, image_min_dim = 640, image_resize_mode ="square", gpu_count = 1):
        self.config = Config(BACKBONE = network_backbone, NUM_CLASSES = 1 +  num_classes,  class_names = class_names, 
       IMAGES_PER_GPU = batch_size, IMAGE_MAX_DIM = image_max_dim, IMAGE_MIN_DIM = image_min_dim, DETECTION_MIN_CONFIDENCE = detection_threshold,
       IMAGE_RESIZE_MODE = image_resize_mode,GPU_COUNT = gpu_count)

        if network_backbone == "resnet101":
            print("Using resnet101 as network backbone For Mask R-CNN model")
        else:
            print("Using resnet50 as network backbone For Mask R-CNN model")
        cf=self.config
        return cf

    def load_pretrained_model(self):
        #load the weights for COCO
        self.model = models.MaskRCNN(mode="training", model_dir = self.model_dir, config = self.config)
        self.model.build(mode="training",config=self.config)
    
    
    def load_dataset(self, dataset):
        labelme_folder1 = os.path.abspath(os.path.join(dataset, "train"))
        #print(labelme_folder1)
        #dir where the converted json files will be saved
        save_json_path1 = os.path.abspath(os.path.join(dataset, "train_json"))
        
      
     
        
        # Training dataset.
        self.dataset_train = Data()
        self.dataset_train.load_data(save_json_path1, labelme_folder1)
        self.dataset_train.prepare()
        dt_train=self.dataset_train
        
        labelme_folder2 = os.path.abspath(os.path.join(dataset, "test"))

        #dir where the converted json files will be saved
        save_json_path2 = os.path.abspath(os.path.join(dataset, "test_json"))
        
        
        
        
        # Training dataset.
        self.dataset_test = Data()
        self.dataset_test.load_data(save_json_path2, labelme_folder2)
        self.dataset_test.prepare()
        dt_test = self.dataset_test
        return dt_train,dt_test

    def visualize_sample(self):
        image_id = np.random.choice(self.dataset_train.image_ids)

        image = self.dataset_train.load_image(image_id)
        mask, class_ids = self.dataset_train.load_mask(image_id)
        bbox = extract_bboxes(mask)


        # Display image and instances
        out = display_box_instances(image, bbox, mask, class_ids, self.dataset_train.class_names)  
        plt.imshow(out)
        plt.axis("off")
        plt.show()
            

        

    def train_model(self, num_epochs, path_trained_models,  layers = "all", augmentation = False):
        if augmentation == False:
            print("No Augmentation")

        else:
            # if augmentation == True:
            #     augmentation = imgaug.augmenters.Sometimes(0.5, [
			#         imgaug.augmenters.Fliplr(0.5),
			#         iaa.Flipud(0.5),
			#         imgaug.augmenters.GaussianBlur(sigma=(0.0, 5.0))
			#         ])
            #     print("Applying Default Augmentation on Dataset")

            # else:
            augmentation = augmentation
            print("Applying Custom Augmentation on Dataset")   

        print('Train %d' % len(self.dataset_train.image_ids), "images")
        print('Validate %d' % len(self.dataset_test.image_ids), "images")

        self.model.train(self.dataset_train, self.dataset_test,models = path_trained_models, augmentation = augmentation, epochs=num_epochs,layers=layers)
                             
        
    def evaluate_model(self, model_path, iou_threshold = 0.5):
        self.model = models.MaskRCNN(mode = "inference", model_dir = os.getcwd(), config = self.config)  
        if os.path.isfile(model_path):
            model_files = [model_path]
             
        if os.path.isdir(model_path):
            model_files = sorted([os.path.join(model_path, file_name) for file_name in os.listdir(model_path)])
        for modelfile in model_files:
            if str(modelfile).endswith(".h5"):
                self.model.load_weights(modelfile, by_name=True)
            APs = []
            #outputs = list()
            for image_id in self.dataset_test.image_ids:                                                                                                                                                                                                                                                                                                                                                                             
                # load image, bounding boxes and masks for the image id
                image, image_meta, gt_class_id, gt_bbox, gt_mask = data_generator.load_image_gt(self.dataset_test, self.config, image_id)
                # convert pixel values (e.g. center)
                scaled_image = models.mold_image(image, self.config)
                # convert image into one sample
                sample = np.expand_dims(scaled_image, 0)
		        # make prediction
                yhat = self.model.detect(sample, verbose=0)
		        # extract results for first sample
                r = yhat[0]
		        # calculate statistics, including AP
                AP, _, _, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'],
                iou_threshold=iou_threshold)
		        # store
                APs.append(AP)
	        # calculate the mean AP across all images
            mAP = np.mean(APs)
            print(modelfile, "evaluation using iou_threshold", iou_threshold, "is", f"{mAP:01f}", '\n')
