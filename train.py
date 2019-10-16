#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import sys
import caffe
import argparse
import numpy as np

def train(initmodel, gpu):
    """
    Train the net.
    """
    caffe.set_mode_gpu()
    caffe.set_device(gpu)
    solver = caffe.AdamSolver('solver.prototxt')
    if initmodel:
        solver.net.copy_from(initmodel)
    solver.step(solver.param.max_iter)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="dataset name.")
    parser.add_argument("--initmodel", help="Init caffemodel.")
    parser.add_argument("--gpu", required=True, type=int, help="Device ids.")
    args = parser.parse_args()

    train(args.initmodel, args.gpu)

