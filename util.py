import time
import matplotlib.pyplot as plt
import os
import csv
from collections import defaultdict
from instance import *
import argparse


def setparser(parser: argparse.ArgumentParser):

    parser.add_argument('--season', type=str,           
                        choices = ['spring', 'summer', 'winter'], default='spring', help='spring, summer, winter')
    parser.add_argument('--dnum', type=int,             
                        default=0,  help='[0,9]: Datanum')
    parser.add_argument('--model', type=str,             
                        choices = ['det', 'ro', 'so'], default = 'det',  help = 'det, ro, so')
    parser.add_argument('--insample', type=int,    
                        choices = [0,1], default=1, help='0: out-of-sample test , 1: in-sample test')
    parser.add_argument('--num_scenario', type=int,     
                        default=100, help='number of scenarios')
    parser.add_argument('--cv', type=float,             
                        default=0.3,  help='[0,1]: cv for uncertainty set')
    parser.add_argument('--cv_scenario', type=float,    
                        default=0.3, help='[0,1]: cv for generate scenarios')
    parser.add_argument('--ctr_style', type=int,    
                        default=0, help='0: sparse, 1: dense')
    parser.add_argument('--gamma', type=float,    
                        default=1, help='[0,1,2,3]')
    
    return parser