# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import dataset_utils
import platform_details
import os
import audio_utils
from matplotlib import pyplot as plt
from numba import jit
import librosa
import librosa.display

DATASET_MEGA = (np.array([]), np.array([]))
DATASET_4 = (np.array([]), np.array([]))
DATASET_TOY = (np.array([]), np.array([]))

def make_dataset(ragas):
    dataset = []
    for r in ragas:
        dataset.append(dataset_utils.get_files_by_genre(r))

    return dataset

def get_audios(genre_files):
    loc_data = genre_files.copy()
    ret = []
    for i in range(len(loc_data)):
        files = dataset_utils.load_audio_files(loc_data[i], n='all')
        if len(files) == 0:
            dir_path = os.path.join(dataset_utils.copy_base, loc_data[i])
            audio_path = os.path.join(dir_path, os.listdir(dir_path)[0])
            audio_utils.cut_file_1min_segments(audio_path)
            files = dataset_utils.load_audio_files(loc_data[i], n='all')
        ret.append(files)
    return ret

def print_dataset_to_tex(genre_files, audios, iter=0):
    home = dataset_utils.copy_base
    file_path = os.path.join(home, 'dataset_tex.txt')
    with open(file_path, 'a') as f:
        for i in range(len(genre_files)):
            name=audios[i][0].strip('out000.wav').strip(home).strip(genre_files[i]).replace('\\', '').replace('_', ' ')
            if dataset_utils.get_Raga_ID(genre_files[i]) == 42:
                print(
                    f"{iter + 1} & {name} & {get_Raga(genre_files[i])} & Kanniks Kannikeswaran" + " \\\ ")
            else:
                print(
                    f"{iter + 1} & {name} & {get_Raga(genre_files[i])} & {dataset_utils.get_Artist(genre_files[i]).replace('_', ' ')}" + " \\\ ")

            string = f"{iter+1} & {name} & {get_Raga(genre_files[i])}" + " \\\ " + "\n"
            # f.write(string)
            iter += 1
    return iter

def load_and_save_genre_audios_to_disk(genre_files):
    loc_data = genre_files.copy()
    for i in range(len(loc_data)):
        dataset_utils.save_file_audio_data(loc_data[i])
        print(f"{i+1} of {len(loc_data)} saved to disk")


def get_Raga(f):
    return dataset_utils.get_Raga(f)

@jit(nopython=True)
def calculate_pcd(data):
    local_data = data.copy()
    pcds = []
    for i in range(len(local_data)):
        raga_pcd = np.array([], dtype='float32')
        for fname in local_data[i]:
            files = dataset_utils.load_audio_files(fname, n='all')
            print(files)

def get_pcd(files, metric='cens'):
    pcd = audio_utils.get_mean_pcd(files,  metric)
    return pcd

def get_chroma(files, variant='cens'):
    chroma = audio_utils.get_chromagram(files, variant)
    return chroma
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data = make_dataset((3, 22, 41, 42))
    audios = (get_audios(data[0]), get_audios(data[1]), get_audios(data[2]), get_audios(data[3]))
    # iter = print_dataset_to_tex(data[0], audios[0])
    # iter = print_dataset_to_tex(data[1], audios[1], iter)
    # iter = print_dataset_to_tex(data[2], audios[2], iter)
    # iter = print_dataset_to_tex(data[3], audios[3], iter)
    labels = [[get_Raga(f) for f in data[0]], [get_Raga(f) for f in data[1]],
              [get_Raga(f) for f in data[2]], [get_Raga(f) for f in data[3]]]
    labels = np.array(labels)
    y = []
    X = []
    for f in data[0]:
        pth = dataset_utils.load_audio_files(f, 1, random=False)
        for c in pth:
            c = c.replace('.wav', '_chromaCENS.npy')
            chroma_stft = np.load(c)
            idx = np.random.choice(len(chroma_stft), 280)
            # chroma_stft = chroma_stft[idx]
            for cstft in chroma_stft:
                X.append(cstft.ravel())
                y.append(dataset_utils.get_Raga_ID(f))
    for f in data[1]:
        pth = dataset_utils.load_audio_files(f, 1, random=False)
        for c in pth:
            c = c.replace('.wav', '_chromaCENS.npy')
            chroma_stft = np.load(c)
            idx = np.random.choice(len(chroma_stft), 280)
            # chroma_stft = chroma_stft[idx]
            for cstft in chroma_stft:
                X.append(cstft.ravel())
                y.append(dataset_utils.get_Raga_ID(f))
    for f in data[2]:
        pth = dataset_utils.load_audio_files(f, 1, random=False)
        for c in pth:
            c = c.replace('.wav', '_chromaCENS.npy')
            chroma_stft = np.load(c)
            idx = np.random.choice(len(chroma_stft), 2)
            # chroma_stft = chroma_stft[idx]
            for cstft in chroma_stft:
                X.append(cstft.ravel())
                y.append(dataset_utils.get_Raga_ID(f))


    X = np.array(X)
    y = np.array(y)
    datasets_folder = os.path.join(dataset_utils.copy_base, 'DATASETS')
    try:
        os.mkdir(datasets_folder)
    except:
        pass
    np.save(os.path.join(datasets_folder,'4raga_cens_X.npy'), X, allow_pickle=False)
    np.save(os.path.join(datasets_folder,'4raga_cens_Y.npy'), y, allow_pickle=False)
    count = 0
    for i in range(len(y)-1):
        if y[i] == 41:
            count += 1
    print(X.shape, y.shape, count)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
