
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
from collections import defaultdict, OrderedDict
from itertools import groupby, izip_longest
import copy, random, pdb


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

numFiles = 2
file_names = os.listdir(path)#List of file names in the directory
file_names = sorted(file_names, key=lambda item: (int(item.partition('.')[0]) if item[0].isdigit() else float('inf'), item))
file_names = file_names[0:numFiles]

##################
#    FUNCTIONS   #
##################

#Obtain chords+notes merged in a single voice
def extractOneVoice(MIDIdata,i):

    # We start taking the first part
    mainInstrument = MIDIdata[i]  # select channel 0
    voices = mainInstrument.getElementsByClass(stream.Voice)
    num_voices = len(voices)

    # Sum up all voices
    melody = mainInstrument.getElementsByClass(stream.Voice)[0]
    for i in range(1, num_voices):
        newVoice = mainInstrument.getElementsByClass(stream.Voice)[i]
        for j in newVoice:
            melody.insert(j.offset, j)

    return melody

#Obtain measures and chords from a MIDI file
def parseMidi(filename):

    MIDIdata = converter.parse(
        path + "/" + filename)  # Parse the MIDI data for separate melody and accompaniment parts.
    # a Score class is obtained. Socre class is a Stream subclass for handling multi-part music.
    par = MIDIdata.parts  # Returns parts of the score. It filters out all other things that might be in a Score object, such as Metadata returning just the Parts.
    num_parts = len(par)
    melody = extractOneVoice(MIDIdata, 0)

    # Add the other parts
    partIndices = range(1, num_parts)
    comp_stream = stream.Voice()
    comp_stream.append([j.flat for i, j in enumerate(MIDIdata) if i in partIndices])
    for i in (partIndices):
        new_part = extractOneVoice(MIDIdata, i)
        for j in new_part:
            melody.insert(j.offset, j)

    melody.removeByClass(note.Rest)
    #melody.removeByClass(note.Note)
    #melody.removeByClass(chord.Chord)

    measures = OrderedDict()
    measureNum = 0
    for part in melody:
        curr_part = stream.Part()
        print(part)
        curr_part.append(part.getContextByClass('Instrument'))
        curr_part.append(part.getContextByClass('MetronomeMark'))
        curr_part.append(part.getContextByClass('KeySignature'))
        curr_part.append(part.getContextByClass('TimeSignature'))
        measures[measureNum] = curr_part
        measureNum += 1

    chords = OrderedDict()
    chordNum = 0
    for part in melody:
        curr_part = stream.Part()
        curr_part.append(part.getContextByClass('Chord'))
        #curr_part.append(part.getContextByClass('Note'))
        chords[chordNum] = curr_part
        chordNum += 1

    return measures,chords

###################
#    READ DATA    #
###################

data = []
for i in range (len(file_names)):
    measures,chords = parseMidi(file_names[1])


