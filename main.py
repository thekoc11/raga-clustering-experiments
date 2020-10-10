import numpy as np
import dataset_utils
import platform_details
import os
import audio_utils
import feat_ops
from matplotlib import pyplot as plt
from numba import jit
import librosa
import librosa.display
import all_scale

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
    for i in range(len(genre_files)):
        name = audios[i][0].strip('out000.wav').strip(home).strip(genre_files[i]).replace('\\', '').replace('_',
                                                                                                            ' ')
        if dataset_utils.get_Raga_ID(genre_files[i]) == 42:
            print(
                f"{iter + 1} & {name} & {get_Raga(genre_files[i])} & Kanniks Kannikeswaran" + " \\\ ")
        else:
            print(
                f"{iter + 1} & {name} & {get_Raga(genre_files[i])} & {dataset_utils.get_Artist(genre_files[i]).replace('_', ' ')}" + " \\\ ")

        iter += 1
    return iter


def load_and_save_genre_audios_to_disk(genre_files):
    loc_data = genre_files.copy()
    for i in range(len(loc_data)):
        dataset_utils.save_file_audio_data(loc_data[i])
        print(f"{i + 1} of {len(loc_data)} saved to disk")


def get_Raga(f):
    return dataset_utils.get_Raga(f)


def get_pcd(files, metric='cens'):
    pcd = audio_utils.get_pcds_unigram(files, metric)
    return pcd


def get_chroma(files, variant='stft'):
    chroma = audio_utils.get_chromagram(files, variant)
    return chroma


# Entry point of the code
if __name__ == '__main__':
    data = make_dataset((3, 22, 41, 42))
    audios = (get_audios(data[0]), get_audios(data[1]), get_audios(data[2]), get_audios(data[3]))
    # Uncomment the following to print compositions used in the and their details in a LATEX compatible format.
    # iter = print_dataset_to_tex(data[0], audios[0])
    # iter = print_dataset_to_tex(data[1], audios[1], iter)
    # iter = print_dataset_to_tex(data[2], audios[2], iter)
    # iter = print_dataset_to_tex(data[3], audios[3], iter)
    # print(data[3])
    # load_and_save_genre_audios_to_disk(data[3])
    labels = [[get_Raga(f) for f in data[0]], [get_Raga(f) for f in data[1]],
              [get_Raga(f) for f in data[2]], [get_Raga(f) for f in data[3]]]
    labels = np.array(labels)

    # The following code defines the data(features extracted from the audio files, to be used for classification

    y = []
    X = []

    for audio in audios[1]:
        chroma = get_chroma(audio, 'stft')
        X.append(chroma)
        print(chroma.shape)


    # X = audio_utils.get_spectrogram(audios[3][5])
    # print(X[7][1].shape)
    # fig, ax = plt.subplots()
    # img = librosa.display.specshow(X[7][1][:, 256:512], x_axis='s', y_axis='chroma')
    # plt.show()

    # The following code saves the extracted features to the disk(.npy format)

    # datasets_folder = os.path.join(r"E:\DATASET", 'pcds_bigram')
    # try:
    #     os.mkdir(datasets_folder)
    # except:
    #     pass
    # np.save(os.path.join(datasets_folder, 'pcds_bigram_042.npy'), X, allow_pickle=False)
    # np.save(os.path.join(datasets_folder,'Y_003_022.npy'), y, allow_pickle=False)
