import numpy as np
import matplotlib.pyplot as plt
import math

epochs = 100

loss = np.load('song_loss_epochs_' + str(epochs) + '.npy')
plt.plot(range(1,epochs+1,1),loss)
plt.title('Loss Function - 2 layers LSTM 128 net')
plt.xlabel('Number of Epochs')
plt.ylabel('Loss Value')
plt.show()