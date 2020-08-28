import numpy as np
import dataset_utils
import platform_details
import os
import audio_utils
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


def get_chroma(files, variant='cens'):
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

    labels = [[get_Raga(f) for f in data[0]], [get_Raga(f) for f in data[1]],
              [get_Raga(f) for f in data[2]], [get_Raga(f) for f in data[3]]]
    labels = np.array(labels)

    # The following code defines the data(features extracted from the audio files, to be used for classification

    y = []
    X = []

    X = audio_utils.get_pcds_bigram(audios[3])

    # X = audio_utils.get_dominant_pitches(audios[3], metric='cens')
    # X = get_pcd(audios[2])

    # X1 = audio_utils.get_scale_sensitive_feats(audios[0])
    # X2 = audio_utils.get_scale_sensitive_feats(audios[1])
    # max_note_events = 0
    # for x in X2:
    #     X1.append(x)
    # print(len(X1), len(X1[0]))

    # feat_dists_X1 = []
    # y1 = []
    # events = [[], []]
    # dists = [[], []]
    #
    # max_in_x1 = max(len(x[1]) for x in X1)
    # max_in_x2 = max(len(x[1]) for x in X2)
    # max_overall = max(max_in_x1, max_in_x2)
    #
    # for x1 in X1:
    #     events[0], events[1], dists[0], dists[1] = x1
    #     feats = (events[1])
    #     feats = np.append(feats, np.zeros(max_overall-len(events[1])), axis=0)
    #     print(f"len current element pitches for 003: {len(events[1])}, {len(feats)}")
    #     feat_dists_X1.append(feats)
    #     y1.append(3)
    #
    # feat_dists_X1 = np.array(feat_dists_X1)
    # y1 = np.array(y1)
    # print(len(feat_dists_X1), len(feat_dists_X1[0]))
    #
    # feat_dists_X2 = []
    # y2 = []
    # for i in range(len(X2)):
    #     events[0], events[1], dists[0], dists[1] = X2[i]
    #     feats = (events[1])
    #     feats = np.append(feats, np.zeros(max_overall-len(events[1])), axis=0)
    #     # print(f"max p in pcd for 042: {dists[0].max()} at index {i}")
    #     feat_dists_X2.append(feats)
    #     y2.append(22)
    #
    # feat_dists_X2 = np.array(feat_dists_X2)
    # y2 = np.array(y2)
    # print(len(feat_dists_X2), len(feat_dists_X2[0]))
    # #
    # X= np.append(feat_dists_X1, feat_dists_X2, axis=0)
    # y = np.append(y1, y2, axis=0)
    # X = feat_dists_X1
    # y = y2
    #
    # int, dur, ons, int_dist, dur_dist = audio_utils.get_note_events_and_distributions(audios[2])
    # print(int_dist.shape, dur_dist.shape)
    # X = int_dist
    # int1, dur1, ons1, int_dist1, dur_dist1 = audio_utils.get_note_events_and_distributions(audios[1])
    # X = np.append(X, int_dist1, axis=0)
    # temp = np.zeros((46, 1))
    # X = np.append(X, temp, axis=1)

    # X = audio_utils.get_data_aug(X[0])
    print(X.shape, X.max(), X.min())#, y.shape)
    # print(X)
    # The following code save the extracted features to the disk(.npy format)
    datasets_folder = os.path.join(r"E:\DATASET", 'pcds_bigram')
    try:
        os.mkdir(datasets_folder)
    except:
        pass
    np.save(os.path.join(datasets_folder, 'pcds_bigram_042.npy'), X, allow_pickle=False)
    # np.save(os.path.join(datasets_folder,'Y_003_022.npy'), y, allow_pickle=False)
