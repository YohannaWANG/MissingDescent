#!/usr/bin/env python3
# -*- coding: utf-8 -*-

n = 20000
d = 3
mu_low = 1.0
mu_high= 2.0
sig_low = 0.0
sig_high = 1.0

seed = 2022
log = False
gpu = 0
import argparse


def parse():
    '''add and parse arguments / hyperparameters'''
    p = argparse.ArgumentParser()
    p = argparse.ArgumentParser(description="Chain Graph Structure Learning from Observational Data")

    p.add_argument('--n', type=int, default=n, help='number of samples')
    p.add_argument('--d', type=int, default=d, help='number of variables')
    p.add_argument('--mu_low', type=int, default=mu_low, help='Lower boundary of the output interval')
    p.add_argument('--mu_high', type=int, default=mu_high, help='Upper boundary of the output interval')
    p.add_argument('--sig_low', type=int, default=sig_low, help='Lower boundary of the output interval')
    p.add_argument('--sig_high', type=int, default=sig_high, help='Upper boundary of the output interval')
    def str2bool(v):
        if isinstance(v, bool): return v
        if v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            return True

    p.add_argument('--seed', type=int, default=seed, help='seed value for randomness')
    p.add_argument("--log", type=str2bool, default=log, help="log on file (True) or print on console (False)")
    p.add_argument('--gpu', type=int, default=gpu, help='gpu number, options: 0, 1, 2, 3, 4, 5, 6, 7')

    return p.parse_args()


import os, inspect, logging, uuid


class Logger():
    def __init__(self, p):
        '''Initialise logger '''

        # setup log checkpoint directory
        current = os.path.abspath(inspect.getfile(inspect.currentframe()))
        Dir = os.path.join(os.path.split(os.path.split(current)[0])[0], "checkpoints")
        self.log = p.log

        # setup log file
        if self.log:
            if not os.path.exists(Dir): os.makedirs(Dir)
            name = str(uuid.uuid4())

            Dir = os.path.join(Dir, name)
            if not os.path.exists(Dir): os.makedirs(Dir)
            p.dir = Dir

            # setup logging
            logger = logging.getLogger(__name__)

            file = os.path.join(Dir, name + ".log")
            logging.basicConfig(format="%(asctime)s - %(levelname)s -   %(message)s", filename=file, level=logging.INFO)
            self.logger = logger

    # function to log
    def info(self, s):
        if self.log:
            self.logger.info(s)
        else:
            print(s)


import torch, numpy as np, random

def setup():
    # parse arguments
    p = parse()
    p.logger = Logger(p)
    D = vars(p)

    # log configuration arguments
    l = [''] * (len(D) - 1) + ['\n\n']
    p.logger.info("Arguments are as follows.")
    for i, k in enumerate(D): p.logger.info(k + " = " + str(D[k]) + l[i])

    # set seed
    s = p.seed
    random.seed(s)
    np.random.seed(s)
    p.rs = np.random.RandomState(s)
    os.environ['PYTHONHASHSEED'] = str(s)

    # set device (gpu/cpu)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(p.gpu)
    p.device = torch.device('cuda') if p.gpu != '-1' and torch.cuda.is_available() else torch.device('cpu')

    return p

