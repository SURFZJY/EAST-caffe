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
from tqdm import tqdm

############ Add argument parser for command line arguments ############
parser = argparse.ArgumentParser(description='Use this script to run EAST-caffe')
parser.add_argument('--input', default='imgs/ic15_train/img_896.jpg', 
                    help='Path to input image for single demo')
parser.add_argument('--input_dir', default='imgs/ic15_test', 
                    help='Path to input image for batch demo')
parser.add_argument('--output_dir', default='results', 
                    help='Path to input image for batch demo')                         
parser.add_argument('--model_def', default='models/mbv3/deploy.prototxt',
                    help='prototxt file')
parser.add_argument('--model_weights', default='snapshot/ic15_iter_32000.caffemodel',
                    help='caffemodel file')   
parser.add_argument('--thr',type=float, default=0.95,
                    help='Confidence threshold.')
parser.add_argument('--nms',type=float, default=0.1,
                    help='Non-maximum suppression threshold.')
parser.add_argument('--infer', default='dnn',
                    help='Inference API, dnn or caffe, recommand dnn inference')                     
parser.add_argument('--gpu',type=int, default=5,
                    help='GPU id (only set when inference API is caffe)')                    
args = parser.parse_args()

############ Utility functions ############
def get_images(img_path):
    '''
    find image files in test data path
    :return: list of files found
    '''
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(img_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    return files
    
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
            center = (0.5 * (p1[0] + p3[0]), 0.5 * (p1[1] + p3[1]))
            detections.append((center, (w, h), -1 * angle * 180.0 / math.pi))
            confidences.append(float(score))

    # Return detections and confidences
    return [detections, confidences]


def resize_image(im, max_side_len=2400):
    '''
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    '''
    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
    resize_h = max(32, resize_h)
    resize_w = max(32, resize_w)
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)
    
    
def single_demo(input, output_dir):
    
    im = cv2.imread(input)    
    im_resized, (rH, rW) = resize_image(im)
    inpHeight, inpWidth, _ = im_resized.shape
    im_name = input.split('/')[-1]
    txt_name = 'res_' + im_name.split('.')[0] + '.txt'
    
    if Inference_API == 'caffe':
        import caffe
        import time
        
        gpu = args.gpu
        caffe.set_device(gpu)  # GPU_id pick
        caffe.set_mode_gpu() # gpu mode

        net = caffe.Net(model_def,      # defines the structure of the model
                        model_weights,  # contains the trained weights
                        caffe.TEST)     # use test mode (e.g., don't perform dropout)
        # new_shape = [im.shape[2], im.shape[0], im.shape[1]]
        # net.blobs['image'].reshape(1, *im.shape)

        mu = np.array([103.94, 116.78, 123.68]) # the mean (BGR) pixel values

        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
        transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
        transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
        transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

        image = caffe.io.load_image(input)
        transformed_image = transformer.preprocess('data', image)
        # copy the image data into the memory allocated for the net
        net.blobs['data'].data[...] = transformed_image
        ### perform classification
        start = time.time()
        output = net.forward() # forward
        elapsed = (time.time() - start) * 1000
        print("CAFFE Inference time: %.2f ms" % elapsed)

        F_score  = output['f_score']
        F_geometry = output['geo_concat']
    
    if Inference_API == 'dnn':

        net = cv2.dnn.readNet(model_weights, model_def, 'caffe')
        blob = cv2.dnn.blobFromImage(im_resized, 1.0, (inpWidth, inpHeight), (123.68, 116.78, 103.94), True, False)
        net.setInput(blob)
        # outs = net.forward(['ScoreMap/score', 'GeoMap'])
        outs = net.forward(['f_score', 'F_geometry'])
        t, _ = net.getPerfProfile()
        print('OPENCV-DNN Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency()))

        F_score = outs[0]
        F_geometry = outs[1]
    
    ## Decode
    boxes, confidences = decode(F_score, F_geometry, confThreshold)

    ## standard-NMS
    print('bbox_number before NMS is %d' % len(boxes))
    indices = cv2.dnn.NMSBoxesRotated(boxes, confidences, confThreshold, nmsThreshold)
    print('bbox_number after NMS is %d' % len(indices))

    txt_save_name = os.path.join(output_dir, txt_name)
    with open(txt_save_name, 'w') as f:
        ## draw bbox
        for i in indices:
            coord = []
            # get 4 corners of the rotated rect
            vertices = cv2.boxPoints(boxes[i[0]])
            # print(vertices)
            # scale the bounding box coordinates based on the respective ratios
            for j in range(4):
                vertices[j][0] /= rW
                vertices[j][1] /= rH
                
            coord.append(int(vertices[1][0]))
            coord.append(int(vertices[1][1]))
            coord.append(int(vertices[2][0]))
            coord.append(int(vertices[2][1]))
            coord.append(int(vertices[3][0]))
            coord.append(int(vertices[3][1]))
            coord.append(int(vertices[0][0]))
            coord.append(int(vertices[0][1]))
            txt_line = ','.join(map(str, coord)) + ', RESULT\n'
            f.write(txt_line)
            
            for j in range(4):
                p1 = (vertices[j][0], vertices[j][1])
                p2 = (vertices[(j + 1) % 4][0], vertices[(j + 1) % 4][1])
                cv2.line(im, p1, p2, (128, 240, 128), 2)
    
    im_save_name = os.path.join(output_dir, im_name)
    cv2.imwrite(im_save_name, im)
    print('result saved at', im_save_name)
    
def batch_demo(input_dir, output_dir):
    imgs = get_images(input_dir)
    for img in tqdm(imgs):
        single_demo(img, output_dir)

############ Parse Args ############
model_def = args.model_def
model_weights = args.model_weights
confThreshold = args.thr
nmsThreshold = args.nms
input = args.input  ## single demo
input_dir = args.input_dir  ## batch demo
output_dir = args.output_dir
Inference_API = args.infer

if __name__ == "__main__":
    single_demo(input, output_dir)
    # batch_demo(input_dir, output_dir)
