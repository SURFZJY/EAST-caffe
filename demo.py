#!/usr/bin/env python
# coding: utf-8

import os
os.environ['GLOG_minloglevel'] = '2' ## ignore the caffe log
import warnings
warnings.filterwarnings('ignore') ## ignore Warning log
import numpy as np
import cv2  ## 3.4.5+ or 4.0 +
import math
import argparse

############ Add argument parser for command line arguments ############
parser = argparse.ArgumentParser(description='Use this script to run EAST-caffe')
parser.add_argument('--input', default='imgs/train450_003.jpg', help='Path to input image')
parser.add_argument('--model_def', default='deploy_fairnas.prototxt',
                    help='prototxt file')
parser.add_argument('--model_weights', default='snapshot/nas_iter_16000.caffemodel',
                    help='caffemodel file')                
# parser.add_argument('--height', type=int, default=256,
                    # help='Preprocess input image by resizing to a specific height. It should be multiple by 32.')  ## 256                   
# parser.add_argument('--width', type=int, default=480,
                    # help='Preprocess input image by resizing to a specific width. It should be multiple by 32.')  ## 480
parser.add_argument('--thr',type=float, default=0.9,
                    help='Confidence threshold.')
parser.add_argument('--nms',type=float, default=0.2,
                    help='Non-maximum suppression threshold.')
parser.add_argument('--gpu',type=int, default=4,
                    help='GPU id (dnn-inference API do not need gpu)')                    
args = parser.parse_args()

############ Utility functions ############
def decode(scores, geometry, scoreThresh):
    detections = []
    confidences = []

    ############ CHECK DIMENSIONS AND SHAPES OF geometry AND scores ############
    assert len(scores.shape) == 4, "Incorrect dimensions of scores"
    assert len(geometry.shape) == 4, "Incorrect dimensions of geometry"
    assert scores.shape[0] == 1, "Invalid dimensions of scores"
    assert geometry.shape[0] == 1, "Invalid dimensions of geometry"
    assert scores.shape[1] == 1, "Invalid dimensions of scores"
    assert geometry.shape[1] == 5, "Invalid dimensions of geometry"
    assert scores.shape[2] == geometry.shape[2], "Invalid dimensions of scores and geometry"
    assert scores.shape[3] == geometry.shape[3], "Invalid dimensions of scores and geometry"
    height = scores.shape[2]
    width = scores.shape[3]
    for y in range(0, height):

        # Extract data from scores
        scoresData = scores[0][0][y]
        x0_data = geometry[0][0][y]
        x1_data = geometry[0][1][y]
        x2_data = geometry[0][2][y]
        x3_data = geometry[0][3][y]
        anglesData = geometry[0][4][y]
        for x in range(0, width):
            score = scoresData[x]

            # If score is lower than threshold score, move to next x
            if(score < scoreThresh):
                continue

            # Calculate offset
            offsetX = x * 4.0
            offsetY = y * 4.0
            angle = anglesData[x]

            # Calculate cos and sin of angle
            cosA = math.cos(angle)
            sinA = math.sin(angle)
            h = x0_data[x] + x2_data[x]
            w = x1_data[x] + x3_data[x]

            # Calculate offset
            offset = ([offsetX + cosA * x1_data[x] + sinA * x2_data[x], offsetY - sinA * x1_data[x] + cosA * x2_data[x]])

            # Find points for rectangle
            p1 = (-sinA * h + offset[0], -cosA * h + offset[1])
            p3 = (-cosA * w + offset[0],  sinA * w + offset[1])
            center = (0.5*(p1[0]+p3[0]), 0.5*(p1[1]+p3[1]))
            detections.append((center, (w,h), -1*angle * 180.0 / math.pi))
            confidences.append(float(score))

    # Return detections and confidences
    return [detections, confidences]

def main():
    
    # Read and store arguments
    model_def = args.model_def
    model_weights = args.model_weights
    # inpHeight = args.height
    # inpWidth = args.width
    confThreshold = args.thr
    nmsThreshold = args.nms
    input = args.input
    
    frame = cv2.imread(input)
    
    height_ = frame.shape[0]
    width_ = frame.shape[1]
    
    ###### support arbitary shape input ###### 
    
    resize_ratio = max(height_, width_) / 512
    height_ /= resize_ratio
    width_ /= resize_ratio
    
    inpHeight = int(height_ // 32 * 32)
    inpWidth = int(width_ // 32 * 32)
    print('Input_shape:' , inpHeight, inpWidth)
    
    ##########################################
    
    rH = frame.shape[0] / float(inpHeight)
    rW = frame.shape[1] / float(inpWidth)
    
    Inference_API = 'dnn'  ## choices=['dnn', 'caffe'], recommand dnn inference
    
    if Inference_API == 'caffe':
        import caffe
        import time
        
        gpu = args.gpu
        caffe.set_device(gpu)  # GPU_id pick
        caffe.set_mode_gpu() # gpu mode
        
        # caffe.set_mode_cpu() #cpu mode

        net = caffe.Net(model_def,      # defines the structure of the model
                        model_weights,  # contains the trained weights
                        caffe.TEST)     # use test mode (e.g., don't perform dropout)
        # new_shape = [frame.shape[2], frame.shape[0], frame.shape[1]]
        # net.blobs['image'].reshape(1, *frame.shape)

        mu = np.array([103.94, 116.78, 123.68]) # the mean (BGR) pixel values

        transformer = caffe.io.Transformer({'image': net.blobs['image'].data.shape})
        transformer.set_transpose('image', (2,0,1))  # move image channels to outermost dimension
        transformer.set_mean('image', mu)            # subtract the dataset-mean value in each channel
        transformer.set_raw_scale('image', 255)      # rescale from [0, 1] to [0, 255]
        transformer.set_channel_swap('image', (2,1,0))  # swap channels from RGB to BGR

        image = caffe.io.load_image(input)
        transformed_image = transformer.preprocess('image', image)

        # copy the image data into the memory allocated for the net
        net.blobs['image'].data[...] = transformed_image

        ### perform classification
        start = time.time()
        output = net.forward() # forward
        elapsed = (time.time() - start) * 1000
        print("CAFFE Inference time: %.2f ms" % elapsed)

        F_score = output['ScoreMap']
        F_geometry = output['GeoMap']
    
    if Inference_API == 'dnn':
        
        net = cv2.dnn.readNet(model_weights, model_def, 'caffe')
        blob = cv2.dnn.blobFromImage(frame, 1.0, (inpWidth, inpHeight), (123.68, 116.78, 103.94), True, False)
        net.setInput(blob)
        # outs = net.forward(['F_score', 'F_geometry']) ## output layer name
        outs = net.forward(['ScoreMap/score', 'GeoMap'])
        t, _ = net.getPerfProfile()
        print('OPENCV-DNN Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency()))

        F_score = outs[0]
        F_geometry = outs[1]
    
    ## Decode
    [boxes, confidences] = decode(F_score, F_geometry, confThreshold)
    
    ## standard-NMS
    indices = cv2.dnn.NMSBoxesRotated(boxes, confidences, confThreshold, nmsThreshold)

    ## draw bbox
    for i in indices:
        # get 4 corners of the rotated rect
        vertices = cv2.boxPoints(boxes[i[0]])
        # scale the bounding box coordinates based on the respective ratios
        for j in range(4):
            vertices[j][0] *= rW
            vertices[j][1] *= rH
        for j in range(4):
            p1 = (vertices[j][0], vertices[j][1])
            p2 = (vertices[(j + 1) % 4][0], vertices[(j + 1) % 4][1])
            cv2.line(frame, p1, p2, (128, 240, 128), 5);
            
    cv2.imwrite('res_demo.jpg', frame)

if __name__ == "__main__":
    main()
    print('Finish~')
