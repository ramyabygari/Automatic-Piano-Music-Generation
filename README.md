## Automatic Piano Music Generation

In this project we generate piano music by using a Recurrent Neural Network architecture. More precisely, we feed the network with a dataset of piano music in the MIDI (Musical Instrument Digital Interface) file format, following a note-by-note approach. Our aim is to get a sound comparable to the original one from the dataset and in order to achieve this goal we made use of an architecture of Recurrent Neural Networks (RNNs) called Long Short-Term Memory (LSTM). The reason of choosing this architecture stems from the ability of these networks to remember states from the past, a property that is especially important when working with time sequences.

For more information check the [report](AutomaticPianoReport.pdf) and for a short overview check the [poster](AutomaticPianoPoster.pdf).

The results can be found in the Results/ folder. Note that this project require further work to achive a similarity to a real piano piece.

## Instrucctions

pip install -r requirements.txt

python model.py pathToData/