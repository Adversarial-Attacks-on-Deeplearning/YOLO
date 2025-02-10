from __future__ import division
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
import argparse
import os 
import os.path as osp
from darknet import Darknet
import pickle as pkl
import pandas as pd
import random

def arg_parse():
    """
    Parse arguements to the detect module
    
    """
    
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
   
    parser.add_argument("--images", dest = 'images', help = 
                        "Image / Directory containing images to perform detection upon",
                        default = "imgs", type = str)
    parser.add_argument("--det", dest = 'det', help = 
                        "Image / Directory to store detections to",
                        default = "det", type = str)
    parser.add_argument("--bs", dest = "bs", help = "Batch size", default = 1)
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help = 
                        "Config file",
                        default = "cfg/yolov3.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help = 
                        "weightsfile",
                        default = "yolov3.weights", type = str)
    parser.add_argument("--reso", dest = 'reso', help = 
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "416", type = str)
    
    return parser.parse_args()


def load_yolo_model(cfgfile,weightsfile, reso ,CUDA = True):
    num_classes = 80
    classes = load_classes("data/coco.names")



    num_classes = 80
    classes = load_classes("data/coco.names")
    #Set up the neural network
    print("Loading network.....")
    model = Darknet(cfgfile)
    model.load_weights(weightsfile)
    print("Network successfully loaded")

    model.net_info["height"] = reso
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0 
    assert inp_dim > 32

    #If there's a GPU availible, put the model on GPU
    if CUDA:
        model.cuda()

    #Set the model in evaluation mode
    model.eval()

    return model, classes, num_classes, inp_dim


def prepare_image(images, inp_dim, batch_size = 1):
    """
    Prepare image for inputting to the neural network. 
    
    Returns a Variable 
    """
    try:
        imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images)]
    except NotADirectoryError:
        imlist = []
        imlist.append(osp.join(osp.realpath('.'), images))
    except FileNotFoundError:
        print ("No file or directory with the name {}".format(images))
        exit()
        


    loaded_ims = [cv2.imread(x) for x in imlist]

    im_batches = list(map(prep_image, loaded_ims, [inp_dim for x in range(len(imlist))]))
    im_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_ims]
    im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)


    leftover = 0
    if (len(im_dim_list) % batch_size):
        leftover = 1

    if batch_size != 1:
        num_batches = len(imlist) // batch_size + leftover            
        im_batches = [torch.cat((im_batches[i*batch_size : min((i +  1)*batch_size,
                            len(im_batches))]))  for i in range(num_batches)] 
    
    return im_batches, im_dim_list, imlist, loaded_ims


def detect_image(model, classes, num_classes, inp_dim, images, batch_size, confidence, nms_thesh,det, CUDA = True):
    """
    Perform detection on images
    return output, imlist, loaded_ims
    output: is a tensor of shape (N, 7) where N is the number of detections
    ind 0: the index of the image in the batch
    ind 1: x coordinate of the top left corner of the predicted bounding box
    ind 2: y coordinate of the top left corner of the predicted bounding box
    ind 3: x coordinate of the bottom right corner of the predicted bounding box
    ind 4: y coordinate of the bottom right corner of the predicted bounding box
    ind 5: objectness score of the prediction
    ind 6: confidence of the prediction
    ind 7: the index of the class detected
    imlist: list of image paths
    loaded_ims: list of loaded images

    """
    
    #Detection phase
    #load the images
    im_batches, im_dim_list, imlist, loaded_ims = prepare_image(images, inp_dim, batch_size)

    #make the detection directory if it doesn't exist
    if not os.path.exists(det):
        os.makedirs(det)

    write = 0


    if CUDA:
        im_dim_list = im_dim_list.cuda()
        

    for i, batch in enumerate(im_batches):
    #load the image 

        if CUDA:
            batch = batch.cuda()

        batch.requires_grad = True  

        with torch.enable_grad():  
            prediction = model(batch, CUDA)
        
        prediction = write_results(prediction, confidence, num_classes, nms_conf = nms_thesh)



        if type(prediction) == int:

            for im_num, image in enumerate(imlist[i*batch_size: min((i +  1)*batch_size, len(imlist))]):
                im_id = i*batch_size + im_num
            continue

        prediction[:,0] += i*batch_size    #transform the atribute from index in batch to index in imlist 

        if not write:                      #If we have't initialised output
            output = prediction  
            write = 1
        else:
            output = torch.cat((output,prediction))

        for im_num, image in enumerate(imlist[i*batch_size: min((i +  1)*batch_size, len(imlist))]):
            im_id = i*batch_size + im_num
            objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
            print("----------------------------------------------------------")
            print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
            print("----------------------------------------------------------")

        if CUDA:
            torch.cuda.synchronize()       
    try:
        output
    except NameError:
        print ("No detections were made")
        exit()

    im_dim_list = torch.index_select(im_dim_list, 0, output[:,0].long())

    scaling_factor = torch.min(416/im_dim_list,1)[0].view(-1,1)


    output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim_list[:,0].view(-1,1))/2
    output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim_list[:,1].view(-1,1))/2



    output[:,1:5] /= scaling_factor

    for i in range(output.shape[0]):
        output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim_list[i,0])
        output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim_list[i,1])
    
    return output, imlist, loaded_ims




def write(x, results ,colors, classes):
    c1 = tuple(map(int, x[1:3]))  # Ensure (x1, y1) are integers
    c2 = tuple(map(int, x[3:5]))  # Ensure (x2, y2) are integers
    img = results[int(x[0])]
    cls = int(x[-1])
    color = random.choice(colors)
    label = "{0}".format(classes[cls])

    # Draw bounding box
    cv2.rectangle(img, c1, c2, color, 2)

    # Add label background
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2_label = (c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4)  # Rename to avoid overwriting `c2`
    cv2.rectangle(img, c1, c2_label, color, -1)

    # Put text on image
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), 
                cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)

    return img

def draw_boxes(loaded_ims,output,imlist,classes, det):
    colors = pkl.load(open("pallete", "rb"))
    list(map(lambda x: write(x, loaded_ims,colors,classes), output))
    det_names = pd.Series(imlist).apply(lambda x: "{}/det_{}".format(det,x.split("/")[-1]))
    list(map(cv2.imwrite, det_names, loaded_ims))
    print("Detection Images are saved in {}".format(det))

    
if __name__ == "__main__":
    args = arg_parse()
    images = args.images
    batch_size = int(args.bs)
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    start = 0
    CUDA = torch.cuda.is_available()

    model, classes, num_classes, inp_dim = load_yolo_model(args.cfgfile,args.weightsfile, args.reso ,CUDA = CUDA)
    output, imlist, loaded_ims = detect_image(model, classes, num_classes, inp_dim, images, batch_size, confidence, nms_thesh,args.det, CUDA = CUDA)
    draw_boxes(loaded_ims,output,imlist,classes, args.det)


