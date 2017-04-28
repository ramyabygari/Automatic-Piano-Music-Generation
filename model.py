
from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import os
from music21 import *

################################
#    CHECK PATH WAS PROVIDED   #
################################

if len(sys.argv) == 2:
    path = sys.argv[1]
else:
    print('Missing data path as first argument')
    sys.exit

###################
#    FILE LIST    #
###################

numFiles = 1
file_names = os.listdir(path)#List of file names in the directory
file_names = sorted(file_names, key=lambda item: (int(item.partition('.')[0]) if item[0].isdigit() else float('inf'), item))
file_names = file_names[0:numFiles]

###################
#    READ DATA    #
###################

data = []
for i in range (len(file_names)):
    MIDIdata = converter.parse(path + "/" + file_names[i])
    MIDIdata.show('text')



