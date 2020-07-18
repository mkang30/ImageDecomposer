#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 13:31:36 2020

@author: minseongkang
"""
import numpy as np
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from PIL import Image
import urllib.request

"""
This class represents the Instance Segmentation model Mask R-CNN
"""
class Segmentor:
    
    def __init__(self):
        #create a predictor
        self._cfg = get_cfg()
        self._predictor = self._makePredictor()
        self._class = MetadataCatalog.get(self._cfg.DATASETS.TRAIN[0]).get("thing_classes")
    
    """
    This method initalizes the model and configuration 
    to return the predictor
    """
    def _makePredictor(self):
        self._cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        self._cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        self._cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        return DefaultPredictor(self._cfg)
    
    """
    This method takes an opencv image and perfroms instance segmentation
    """
    def predict(self, image):
        return self._predictor(image)

    """
    This method takes an output of the model and returns the segmentation
    map of the image
    """
    def segMap(self, image,output):
        v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(self._cfg.DATASETS.TRAIN[0]), scale=1.2)
        v = v.draw_instance_predictions(output["instances"].to("cpu"))
        r = Image.fromarray(v.get_image()[:,:,::-1])
        return r
    """
    This method takes an output of the model and returns an array of images
    of key objects in the input image.
    """
    def decompose(self, image, output):
        r = dict()
        count = dict()
        for i in range(len(output["instances"].pred_boxes)):
            box = output["instances"].pred_boxes[i].tensor.numpy()[0]
            dim = (box[0],box[1],box[2],box[3])
            mask = output["instances"].pred_masks[i]
            image2 = image[:,:,::-1].copy()
            for j in range(len(image)):
                for k in range(len(image[j])):
                    if mask[j][k]==False:
                        for l in range(len(image2[j][k])):
                            image2[j][k][l]=255
            pic = Image.fromarray(image2)
            pic = pic.crop(dim)
            pic = self._transparent(pic)
            cl = self._class[output["instances"].pred_classes[i]]
            if cl in count:
                count[cl]+=1
            else:
                count[cl]=1
            r[cl+str(count[cl])]=pic
        return r
    """
    Input: PIL image
    Output: PIL image with transparent background
    """
    def _transparent(self,image):
        r = image.convert("RGBA")
        pixels = r.getdata()
        newPixels = []
        for i in pixels:
            if i[0]==255 and i[1]==255 and i[2]==255:
                newPixels.append((255,255,255,0))
            else:
                newPixels.append(i)
        r.putdata(newPixels)
        return r




       

        

