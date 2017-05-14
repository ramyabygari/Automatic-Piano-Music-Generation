#/Users/albertbou/Automatic-Piano-Music-Generation/Data

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
import copy, random, pdb

# pdb.set_trace() -> breakpoint

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

##################
#    FUNCTIONS   #
##################

#Plays song in real time -> requires pygame
def playSong(MIDIdata): #input is class part or score
    sp = midi.realtime.StreamPlayer(MIDIdata)
    sp.play()

#Obtain chords+notes merged in a single voice
def extractOneVoice(MIDIdata,i):
    # We start taking the first part
    mainInstrument = MIDIdata[i]  # select channel 0
    voices = mainInstrument.getElementsByClass(stream.Voice)
    num_voices = len(voices)
    # Sum up all voices in case there is more than one
    melody = mainInstrument.getElementsByClass(stream.Voice)
    if len(melody) > 0:
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
    melody.removeByClass(note.Note)
    #melody.removeByClass(chord.Chord)

    measures = OrderedDict()
    measureNum = 0
    for part in melody:
        curr_part = stream.Voice()
        curr_part.insert(part.offset,part.getContextByClass('Instrument'))
        curr_part.insert(part.offset,part.getContextByClass('MetronomeMark'))
        curr_part.insert(part.offset,part.getContextByClass('KeySignature'))
        curr_part.insert(part.offset,part.getContextByClass('TimeSignature'))
        measures[measureNum] = curr_part
        measureNum += 1

    chords = OrderedDict()
    chordNum = 0
    for part in melody:
        curr_part = stream.Voice()
        if part.getContextByClass('Chord') != None:
            curr_part.insert(part.offset, part.getContextByClass('Chord'))
        if part.getContextByClass('Note') != None:
            curr_part.insert(part.offset,part.getContextByClass('Note'))
        chords[chordNum] = curr_part
        chordNum += 1

    return measures,chords

#Creates a MIDI stream from chords/measures -> at the moment only chords used for reconstruction
def generateMIDI(chords,measures):
    lensong = len(chords)
    song = stream.Voice()
    for i in range(lensong):
        try:
            #song.insertIntoNoteOrChord(chords[i].offset, chords[i], chordsOnly=False)
            chords[i].offset  = i
            song.append(chords[i])
        except exceptions21.StreamException:
            print('warning: Note or Chord is already found in this Stream! solve that at some point!')

    voices = song.getElementsByClass(stream.Voice)
    num_voices = len(voices)
    song2 = stream.Voice()
    # Sum up all voices
    for i in range(0, num_voices):
        newVoice = song.getElementsByClass(stream.Voice)[i]
        t = 0
        for j in newVoice:
            try:
                song2.insert(j.offset, j)
            except exceptions21.StreamException:
                print('warning: Note or Chord is already found in this Stream! solve that at some point!')
        t += 1
    return song

# helper function to sample an index from a probability array -> allows to have variability
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# Return unique chords of the dataset
def getUniqueChords(data):
    chords = []
    chordNames = []
    for chord in data:
        if not (chord.fullName in chordNames):
            chords.append(chord)
            chordNames.append(chord.fullName)

    return chords

###################
#    READ DATA    #
###################

data = []
merged_dict = OrderedDict()
for i in range (len(file_names)):
    measures,chords = parseMidi(file_names[i])

    #song = generateMIDI(chords,measures)
    #playSong(song)
    values = []
    for i in range(len(chords)):
        values.append(chords[i][0])
        #if len(chords[i]) > 1:   -> If notes also added, more than 1 thing at the same time!!
        #    values.append(chords[i][1])
    data += values
    len(data)

########################################
#    ORGANIZE CHORDS IN THE DATASET    #
########################################

print('total chords:', len(data))
vals = getUniqueChords(data)
val_indices = dict((v, i) for i, v in enumerate(vals))
indices_val = dict((i, v) for i, v in enumerate(vals))

######################
#    VECTORIZATION   #
######################

print('Vectorization...')
maxlen = 40
step = 3
pieces = []
next_chords = []

for i in range(0, len(data) - maxlen, step): #"sound frames", overlapping of 40-3 characters
    pieces.append(data[i: i + maxlen])
    next_chords.append(data[i + maxlen])

print('nb sequences:', len(pieces))

X = np.zeros((len(pieces), maxlen, len(vals)), dtype=np.bool)
y = np.zeros((len(pieces), len(vals)), dtype=np.bool)

for i, piece in enumerate(pieces):
    for t, chord in enumerate(piece):
        X[i, t, val_indices[chord]] = 1
    y[i, val_indices[next_chords[i]]] = 1

#####################################
#    BUILD THE MODEL: SIMPLE LSTM   #
#####################################

print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(vals))))
model.add(Dense(len(vals)))
model.add(Activation('softmax'))
optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

#################
#    TRAINING   #
#################

for iteration in range(1, 10):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X, y,
              batch_size=128,
              epochs=1)

    start_index = random.randint(0, len(data) - maxlen - 1)#valor random entre 0 y 200287-40-1

    for diversity in [1.2]:#0.2, 0.5, 1.0, 1.2
        print()
        print('----- diversity:', diversity)

        generated = list()
        seed = data[start_index: start_index + maxlen]#frase inicial son los 40 caracteres despues de start_index
        #generated += seed
        print('----- Generating with seed: "')
        print(seed)
        #print(generated)

        for i in range(30):# 40 predicted chords
            x = np.zeros((1, maxlen, len(vals)))
            for t, chord in enumerate(seed):
                x[0, t, val_indices[chord]] = 1.# One-hot representation of the randomly selected sentence

            preds = model.predict(x, verbose=0)[0]# Output is a prob vector of 59 positions
            next_index = sample(preds, diversity)# Sample an index from the probability array
            next_chord = indices_val[next_index]# Identifies the character

            generated += next_chord
            seed.append(next_chord)
            del seed[0]

            print(next_chord)
            sys.stdout.flush()

        print()

########################
#    PLAY GENERATED    #
########################

a = generateMIDI(generated,list())
#playSong(a)
