from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.layers import Dropout
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
from keras.callbacks import Callback
import numpy as np
import matplotlib.pyplot as plt
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

numFiles = 6
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
    print("processing", filename)
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

    measures = []
    measureNum = 0
    for part in melody:
        curr_part = stream.Voice()
        curr_part.insert(part.offset,part.getContextByClass('Instrument'))
        curr_part.insert(part.offset,part.getContextByClass('MetronomeMark'))
        curr_part.insert(part.offset,part.getContextByClass('KeySignature'))
        curr_part.insert(part.offset,part.getContextByClass('TimeSignature'))
        measures.append(curr_part)
        measureNum += 1

    chords = []
    chordNum = 0
    for part in melody:
        curr_part = stream.Voice()
        if part.getContextByClass('Chord') != None:
            curr_part.insert(part.offset, part.getContextByClass('Chord'))
        if part.getContextByClass('Note') != None:
            curr_part.insert(part.offset,part.getContextByClass('Note'))
        chords.append(curr_part)
        chordNum += 1

    return measures,chords

#Creates a MIDI stream from chords/measures -> at the moment only chords used for reconstruction
def generateMIDI(chords):
    lensong = len(chords)
    part = stream.Part()
    song = stream.Voice()
    for i in range(lensong):
        notas = []
        if (len(chords[i][0]) != 0):
            for j in chords[i][0][0]:
                a = j.nameWithOctave
                notas.append(note.Note(a))
            nextchord = chord.Chord(notas)
            song.insert(i,nextchord)
        for n in chords[i][1]:
            song.insert(i, note.Note(n.nameWithOctave))

    part.insert(0, song)

    for i in range(lensong):
        #print(chords[i][2].ratioString)
        #print(chords[i][1].name)
        part.insert(i, key.KeySignature(chords[i][2].sharps))
        part.insert(i, meter.TimeSignature(chords[i][3].ratioString))

    return part

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
    info = []
    for chord in data:
        newinfo = generateKey(chord)
        if not (newinfo in info):
            info.append(newinfo)
            chords.append(chord)
    return chords, info


def generateKey(object):
    notesName = ""
    chordsName = ""
    for n in object[0]:
        chordsName += n.fullName
    for n in object[1]:
        notesName += n.nameWithOctave
    return chordsName + notesName + object[2].name + object[3].ratioString

###################
#    READ DATA    #
###################

data = []

for i in range (len(file_names)):
    measures,chords = parseMidi(file_names[i])
    values = []
    for j in range(len(chords)):
        timestep = []
        chordList = []
        noteList = []
        for t in chords[j]:
            if t.isChord:
                chordList.append(t)
            elif t.isNote:
                noteList.append(t)
        timestep.append(chordList)
        timestep.append(noteList)
        timestep.append(measures[j][2])
        timestep.append(measures[j][3])
        values.append(timestep)
    data += values
    len(data)

#data[0][0].fullName
#data[0][1].name
#data[0][2].ratioString

########################################
#    ORGANIZE CHORDS IN THE DATASET    #
########################################

print('total chords:', len(data))
vals, info = getUniqueChords(data)

val_indices = dict((inf, i) for i, inf in enumerate(info))
indices_val = dict((i, v) for i, v in enumerate(vals))

print('dicctionary length:', len(val_indices))

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
    for t, acorde in enumerate(piece):
        X[i, t, val_indices[generateKey(acorde)]] = 1
    y[i, val_indices[generateKey(next_chords[i])]] = 1

######################################
#    BUILD THE MODEL: 2 LAYER-LSTM   #
######################################

print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(vals)),return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(len(vals)))
model.add(Activation('softmax'))
#optimizer = RMSprop(lr=0.01)
optimizer = RMSprop()
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

#################
#    TRAINING   #
#################


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        pass

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

num_epochs = 10
history = LossHistory()
history.losses = []
for epoch in range(1, num_epochs+1):
    print()
    print('-' * 50)
    print('epoch', epoch)
    model.fit(X, y,
              batch_size=128,
              epochs=1,
              callbacks=[history])

    start_index = random.randint(0, len(data) - maxlen - 1)

    for diversity in [0.2, 0.5, 1.0, 1.2]:#0.2, 0.5, 1.0, 1.2
        print()
        print('----- diversity:', diversity)

        generated = list()
        seed = data[start_index: start_index + maxlen]#frase inicial son los 40 caracteres despues de start_index
        #generated += seed
        print('----- Generating with seed: "')
        #print(seed)
        #print(generated)

        for i in range(40):# 40 predicted chords
            x = np.zeros((1, maxlen, len(vals)))
            for t, acorde in enumerate(seed):
                x[0, t, val_indices[generateKey(acorde)]] = 1.# One-hot representation of the randomly selected sentence

            preds = model.predict(x, verbose=0)[0]# Output is a prob vector of 59 positions
            next_index = sample(preds, diversity)# Sample an index from the probability array
            next_chord = indices_val[next_index]# Identifies the character

            generated.append(next_chord)
            seed.append(next_chord)
            del seed[0]

            # print(next_chord)
            sys.stdout.flush()

        if (epoch % 5) == 0 or epoch == 1 or epoch == num_epochs:
            generated = generateMIDI(generated)
            name = 'song_epoch' + str(epoch) + '_diversity' + str(diversity) + '.mid'
            fp = generated.write('midi', fp=name)
            # plt.plot(history.losses)
            # plt.xlabel('Epochs')
            # plt.ylabel('Loss')
            # plt.title('Loss function - 2 layers LSTM 128 net')
            # # plt.show()
            # plt.savefig('song_epochs_' + str(epoch) +'.png')
            np.save('song_loss_epochs_' + str(epoch) +'.npy', np.array(history.losses))

        print()

plt.plot(history.losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss function - 2 layers LSTM 128 net')
plt.show()
# plt.savefig('song_epochs_' + str(num_epochs) +'.png')
np.save('song_loss_epochs_' + str(num_epochs) +'.npy', np.array(history.losses))

########################
#    PLAY GENERATED    #
########################

#playSong(a)
