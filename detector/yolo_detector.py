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


def load_yolo_model(cfgfile,weightsfile,classes_file, reso ,CUDA = True):

    num_classes = 80
    classes = load_classes(classes_file)
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



def detect_image(model, classes, num_classes, inp_dim, im_batches, im_dim_list, imlist, loaded_ims, batch_size, confidence, nms_thesh, CUDA=True):
    """
    Perform detection on images (with aspect ratio distortion support)
    """
    write = 0
    output = None

    if CUDA:
        im_dim_list = im_dim_list.cuda()

    for i, batch in enumerate(im_batches):
        if CUDA:
            batch = batch.cuda()

        with torch.no_grad():  # Disable gradient calculation
            prediction = model(batch, CUDA)
        
        prediction = write_results(prediction, confidence, num_classes, nms_conf=nms_thesh)

        if isinstance(prediction, int):  # No detections
            for im_num, image in enumerate(imlist[i*batch_size: min((i+1)*batch_size, len(imlist))]):
                im_id = i*batch_size + im_num
                print(f"Image {im_id}: No detections")
            continue

        # Map detection indices to original image order
        prediction[:, 0] += i * batch_size

        # Aggregate predictions
        output = prediction if not write else torch.cat((output, prediction))
        write = 1

        # Print detected objects
        for im_num, image in enumerate(imlist[i*batch_size: min((i+1)*batch_size, len(imlist))]):
            im_id = i*batch_size + im_num
            objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
            print("-"*58)
            print(f"{'Objects Detected:':20s} {' '.join(objs)}")
            print("-"*58)

    if output is None:
        print("No detections were made")
        return output, imlist, loaded_ims

    # ========== Critical Scaling Section ========== #
    # Get original dimensions for each detection
    im_dim_list = torch.index_select(im_dim_list, 0, output[:, 0].long())

    # Calculate independent scaling factors
    width_scale = (im_dim_list[:, 0] / inp_dim).view(-1, 1)  # original_w / resized_w
    height_scale = (im_dim_list[:, 1] / inp_dim).view(-1, 1)  # original_h / resized_h

    # Scale bounding box coordinates
    output[:, [1, 3]] *= width_scale   # x1, x2
    output[:, [2, 4]] *= height_scale  # y1, y2

    # Clamp coordinates to image boundaries
    min_val = torch.tensor(0.0, device=output.device)  # Tensor for min (0.0)
    max_width = im_dim_list[:, 0].view(-1, 1)
    max_height = im_dim_list[:, 1].view(-1, 1)

    output[:, [1, 3]] = torch.clamp(output[:, [1, 3]], min=min_val, max=max_width)
    output[:, [2, 4]] = torch.clamp(output[:, [2, 4]], min=min_val, max=max_height)
    # ============================================== #

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

def draw_boxes(loaded_ims, output, imlist, classes, det, im_name_prefix=""):
    if output is None:
        print("No detections were made")
        return
    
    # Create detection directory if it doesn't exist
    if not os.path.exists(det):
        os.makedirs(det)

    colors = pkl.load(open("pallete", "rb"))
    
    # Draw bounding boxes on images
    list(map(lambda x: write(x, loaded_ims, colors, classes), output))
    
    # Generate new filenames with im_name prefix
    det_names = pd.Series(imlist).apply(
        lambda x: os.path.join(det, f"{im_name_prefix}_{os.path.basename(x)}")
    )
    
    # Save images with new names
    list(map(cv2.imwrite, det_names, loaded_ims))
    print(f"Detection images saved in {det} with prefix: {im_name_prefix}_")

    
if __name__ == "__main__":
    args = arg_parse()
    images = args.images
    batch_size = int(args.bs)
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    start = 0
    CUDA = torch.cuda.is_available()

    model, classes, num_classes, inp_dim = load_yolo_model(args.cfgfile, args.weightsfile, args.reso, CUDA=CUDA)
    
    # Prepare images: ADD THIS STEP
    im_batches, im_dim_list, imlist, loaded_ims = prepare_image(images, inp_dim, batch_size)
    
    # Correctly pass parameters to detect_image
    output, imlist, loaded_ims = detect_image(
        model, classes, num_classes, inp_dim,
        im_batches, im_dim_list, imlist, loaded_ims,
        batch_size, confidence, nms_thesh, CUDA=CUDA
    )
    
    draw_boxes(loaded_ims, output, imlist, classes, args.det)