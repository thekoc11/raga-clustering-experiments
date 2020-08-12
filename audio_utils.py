from dataset_utils import copy_base, get_metadata
import numpy as np
import librosa
import matplotlib.pyplot as plt
import ffmpeg
import numba

PITCHES = {}
GLOBAL_SAMPLING_RATE = 22050
def get_one_min(sr):
    return sr * 60

def get_mean_pcd(files, metric='cens'):
    N = files.shape[0]
    pitch_class = {i:0 for i in range(12)}
    retVal = np.array([], dtype='float32')
    for j in range(len(files)):
        y, sr = librosa.load(files[j])
        pcd_vals = _calculate_pcd_single(y, sr, metric)
        for i in range(12):
            pitch_class[i] += pcd_vals[i]
        print(f"piece {j+1} of {N} pieces done")
    for i in range(12):
        pitch_class[i] /= N
        retVal = np.append(retVal, pitch_class[i])
    return retVal



def  get_chromagram(files, metric='cens'):
    Chroma  = []
    for filename in files:
        y = np.load(filename.replace('.wav', '.npy'))
        sr = GLOBAL_SAMPLING_RATE
        y_harm, y_perc = librosa.effects.hpss(y)
        if metric == 'cens':
            chroma = librosa.feature.chroma_cens(y_harm, sr)
            final_shape = int(get_one_min(sr)/512) + 2
        elif metric == 'stft':
            chroma = librosa.feature.chroma_stft(y_harm, sr, n_fft=512, hop_length=128)
            final_shape = int(get_one_min(sr) / 128) + 2
        else:
            chroma = librosa.feature.chroma_cqt(y_harm, sr)
            final_shape = int(get_one_min(sr) / 512) + 2
        if chroma.shape[1] < final_shape:
            chroma = np.hstack([chroma, np.zeros((chroma.shape[0], (final_shape - chroma.shape[1])))])
        Chroma.append(chroma)
    Chroma = np.array(Chroma)
    return Chroma

def _calculate_pcd_single(sample, sr, metric='cens'):
    PITCHES = {i:0 for i in range(12)}
    y_harm, y_perc = librosa.effects.hpss(sample)
    if metric == 'cens':
        chroma = librosa.feature.chroma_cens(y_harm, sr=sr)
    elif metric == 'stft':
        chroma = librosa.feature.chroma_stft(y_harm, sr, n_fft=512, hop_length=128)
    else:
        chroma = librosa.feature.chroma_cqt(y_harm, sr)

    pitches = chroma.T.argmax(axis=1)
    for p in pitches:
        for key, val in PITCHES.items():
            if p==key:
                PITCHES[key] += 1

    for i in range(chroma.shape[0]):
        PITCHES[i] /= pitches.shape[0]

    return PITCHES

def cut_file_1min_segments(filename):
    out = ffmpeg.input(filename).output(filename=filename.replace('.mp3', '_out%03d.wav'), f='segment', segment_time=60,
                                     c='copy')
    out.run()

if __name__ == '__main__':
    files = librosa.util.find_files(copy_base)
    fname = np.random.choice(files)

    print(fname)
    cut_file_1min_segments(fname)
    # signal, sr = librosa.load(fname)
    # three_mins = 3*get_one_min(sr)
    # one_min = get_one_min(sr)
    # signal = signal[-three_mins:-one_min]
    # raga = get_metadata_windows(fname)
    # print(raga)
    # pcd_X, pcd_Y = calculate_pcd(signal, sr)
    # plt.plot(pcd_X, pcd_Y, label=raga)
    # plt.show()