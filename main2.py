
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

y, sr = librosa.load(r"C:\Users\theko\Documents\Dataset\022035001\Tukiya_Tiruvadi.mp3",sr=22050)

cqt = librosa.cqt(y, sr, fmin=librosa.note_to_hz("C2"), n_bins=48)

c = np.abs(cqt)

# fig, ax = plt.subplots()
#
# img = librosa.display.specshow(librosa.amplitude_to_db(c, ref=np.max), sr=sr, x_axis='time', y_axis='cqt_note', ax=ax)
#
# ax.set_title('Constant-Q power spectrum')
#
# fig.colorbar(img, ax=ax, format="%+2.0f dB")
i = np.argmax(c, axis=0)

notes = np.zeros((48, ))
for i_ in i:
    notes[i_] += 1

notes_X = np.linspace(48, 96, 48, dtype='int32')
plt.plot(notes_X, notes)
plt.show()
