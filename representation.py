from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
import sys
import os
from music21 import *
from collections import defaultdict, OrderedDict
import copy, random, pdb
from sklearn.manifold import TSNE

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

numFiles = 10
numGenerated = 4
numTrained = numFiles - numGenerated
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
    print("finish parsing")
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
        #curr_part.insert(part.offset,part.getContextByClass('Instrument'))
        #curr_part.insert(part.offset,part.getContextByClass('MetronomeMark'))
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


# Return unique chords of the dataset
def getUniqueChords(data):
    chords = []
    info = []
    for song in data:
        for chord in song:
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
        timestep.append(measures[j][0])
        timestep.append(measures[j][1])
        values.append(timestep)
    print("finish processing (our code)", file_names[i])
    data.append(np.array(values))

#data[0][0].fullName
#data[0][1].name
#data[0][2].ratioString

########################################
#    ORGANIZE CHORDS IN THE DATASET    #
########################################

print('total songs:', len(data))
vals, info = getUniqueChords(data)

val_indices = dict((inf, i) for i, inf in enumerate(info))
indices_val = dict((i, v) for i, v in enumerate(vals))

print('dicctionary length:', len(val_indices))


#X = np.zeros((len(data), len(vals)), dtype=np.bool)
X = np.zeros((len(data), len(vals)))

for i, piece in enumerate(data):
    for t, acorde in enumerate(piece):
        X[i, val_indices[generateKey(acorde)]] += 1
    #Normalize
    print(piece.shape[0])
    if (piece.shape[0]):
        X[i, :] /= piece.shape[0]
  

model = TSNE(n_components=2, random_state=0)
np.set_printoptions(suppress=True)
vis_data = model.fit_transform(X)

print(vis_data)
np.save('vis_data.npy', vis_data)

vis_x = vis_data[:, 0]
vis_y = vis_data[:, 1]

n = [0.2, 0.5, 1.0, 1.2]

fig, ax = plt.subplots()
ax.scatter(vis_x, vis_y, c=np.append(['blue'] * numTrained, ['green'] * numGenerated), cmap=plt.cm.get_cmap("jet", 10))
for i, txt in enumerate(n):
    ax.annotate(txt, (vis_x[i+numTrained],vis_y[i+numTrained]))

red_patch = mpatches.Patch(color='blue', label='Training data data')
blue_patch = mpatches.Patch(color='green', label='Generated data')
plt.legend(handles=[red_patch, blue_patch])
plt.title('TSNE Data visualization')
#plt.show()
plt.savefig('vis_data.png')

