from dataset_utils import copy_base, get_metadata
import numpy as np
import librosa
import matplotlib.pyplot as plt
import ffmpeg
import numba
from Viewpoints import Viewpoints

PITCHES = {i: 0 for i in range(12)}
GLOBAL_SAMPLING_RATE = 22050
def get_one_min(sr):
    return sr * 60

def get_mean_pcd(files, metric='cens'):
    N = files.shape[0]
    pitch_class = {i:0 for i in range(12)}
    retVal = np.array([], dtype='float64')
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

def get_CQT(files):
    N = files.shape[0]
    CQT = []
    for i in range(len(files)):
        y, sr = librosa.load(files[i])
        y_harm, y_perc = librosa.effects.hpss(y)
        cqt = librosa.cqt(y_harm, sr=sr, fmin=librosa.note_to_hz('C2'), n_bins=60)
        CQT.append(cqt)
        print(f"file {i+1} of {N} files done")
    return CQT

def get_STFT(files):
    N = files.shape[0]
    STFT = []
    for i in range(len(files)):
        y, sr = librosa.load(files[i])
        y_harm, y_perc = librosa.effects.hpss(y)
        cqt = librosa.stft(y_harm)
        STFT.append(cqt)
        print(f"file {i} of {N} files done")
    return STFT



def get_chromas(file, metric='cens'):
    if metric == 'stft':
        chroma_path = file[0].replace('out000.wav', 'chromaSTFT.npy')
    else:
        chroma_path = file[0].replace('.wav', '_chromaCENS.npy')
    chromas = np.load(chroma_path)
    return chromas


def get_pcds_unigram(files, metric='cens'):
    pcds = []
    for file in files:
        chromas = get_chromas(file, metric)
        for chroma in chromas:
            pitches = chroma.T.argmax(axis=1)
            pitch_counts = np.zeros(12)
            for p in pitches:
                pitch_counts[p] += 1
            pitch_counts /= len(pitches)
            pcds.append(pitch_counts)
    pcds = np.array(pcds)
    # pcds = np.sort(pcds)
    return pcds



def index_to_pitch(index):
    PID = {0:'C', 1:'C#', 2:'D', 3:'E!', 4:'E', 5:'F', 6:'F#', 7:'G', 8:'G#', 9:'A', 10:'B!', 11:'B'}
    return PID[index]

def _default_bigram_dict():
    Classes= {}
    for i in range(12):
        for j in range(12):
            ind = index_to_pitch(i) + index_to_pitch(j)
            Classes[ind] = 0
    return Classes

def get_pcds_bigram(files):
    PCDs = []
    for file in files:
        chromas = get_chromas(file, metric='stft')
        for chroma in chromas:
            pcds = []
            pitches = chroma.T.argmax(axis=1)
            classes = _default_bigram_dict()
            for i in range(1, len(pitches)):
                ind = index_to_pitch(pitches[i-1]) + index_to_pitch(pitches[i])
                classes[ind] += 1
            for (key, value) in classes.items():
                pcds.append(value/((len(pitches) - 1)))
            pcds = np.array(pcds)
            PCDs.append(pcds)
    PCDs = np.array(PCDs)
    return PCDs

def get_int_dist(ints):
    ints += 11
    ret = np.zeros(23)
    for i in ints:
        ret[i] += 1
    ret /= len(ints)
    return ret
def get_dur_dist(dur, ons):
    ret = np.zeros(5000)
    for i in dur:
        ret[i] += 1
        if i > 1:
            ret[i-1] -= 1
    events = 0
    for i in ons:
        if int(i) == 1:
            events += 1
    if events != 0.:
        ret /= events
    else:
        ret = np.zeros(5000)
    return ret

def get_dominant_pitches(files, metric='stft'):
    Dom_pitches = []
    for file in files:
        chromas = get_chromas(file, metric=metric)
        for chroma in chromas:
            dom_pitches = _calculate_dom_pitches(chroma.T.argmax(axis=1))
            Dom_pitches.append(dom_pitches)
    Dom_pitches = np.array(Dom_pitches)
    return Dom_pitches


def _calculate_dom_pitches(pitches):
    dom_pitches = [[], [], [], [], [], [], [], [], [], [], [], []]
    dom_pitches[0] = pitches
    dom_pitches[1] = ((dom_pitches[0] + 1) % 12)
    dom_pitches[2] = ((dom_pitches[0] + 2) % 12)
    dom_pitches[3] = ((dom_pitches[0] + 3) % 12)
    dom_pitches[4] = ((dom_pitches[0] + 4) % 12)
    dom_pitches[5] = ((dom_pitches[0] + 5) % 12)
    dom_pitches[-1] = (dom_pitches[0] - 1) % 12
    dom_pitches[-2] = (dom_pitches[0] - 2) % 12
    dom_pitches[-3] = (dom_pitches[0] - 3) % 12
    dom_pitches[-4] = (dom_pitches[0] - 4) % 12
    dom_pitches[-5] = (dom_pitches[0] - 5) % 12
    dom_pitches[-6] = (dom_pitches[0] - 6) % 12

    return np.array(dom_pitches)

## Deprecated
def get_note_events_and_distributions(files):
    Ints = []
    Durs = []
    Onsets = []
    Ints_Dist = []
    Durs_Dist = []
    for file in files:
        chromas = get_chromas(file, metric='cens')
        for chroma in chromas:
            pitches = chroma.T.argmax(axis=1)
            dur = np.ones_like(pitches)
            int = np.zeros_like(pitches)
            onset = np.zeros_like(dur)
            for i in range(1, len(pitches)):
                shift = pitches[i] - pitches[i-1]
                int[i] = shift
                if shift == 0:
                    dur[i] = dur[i-1] + 1
                if dur[i] == 1:
                    onset[i] = 1
            int_dist = get_int_dist(int)
            dur_dist = get_dur_dist(dur, onset)

            Ints.append(int)
            Durs.append(dur)
            Onsets.append(onset)
            Ints_Dist.append(int_dist)
            Durs_Dist.append(dur_dist)
    Ints = np.array(Ints)
    Durs = np.array(Durs)
    Onsets = np.array(Onsets)
    Ints_Dist = np.array(Ints_Dist)
    Durs_Dist = np.array(Durs_Dist)
    return (Ints, Durs, Onsets, Ints_Dist, Durs_Dist)


def get_scale_sensitive_feats(files):
    Features = []
    for file in files:
        chromas = get_chromas(file)
        feats = [[], [], [], [], []]
        for ind in range(len(chromas)):
            chroma = _add_rests(chromas[ind])
            vp = Viewpoints(chroma)
            # print(vp.pitches.shape)
            # if vp.get_viewpoint('pitch_contour')[0] != None:
            #     pit = vp.get_viewpoint('pitch')
            #     feats[0] = pit
            #     int = vp.get_viewpoint('interval')
            #     # int = np.append(np.array(feats[0]), vp.get_viewpoint('interval'), axis=0)
            #     feats[1] = int
            #     dur = vp.get_viewpoint('duration')
            #     # dur = np.append(np.array(feats[1]), vp.get_viewpoint('duration'), axis=0)
            #     feats[2] = dur
            #     ons = vp.get_viewpoint('onsets')
            #     # ons = np.append(np.array(feats[2]), vp.get_viewpoint('onsets'), axis=0)
            #     feats[3] = ons
            #     ioi = vp.get_viewpoint('ioi')
            #     # ioi = np.append(np.array(feats[3]), vp.get_viewpoint('ioi'), axis=0)
            #     feats[4] = ioi
            #     # pcontour = vp.get_viewpoint('pitch_contour')
            #     # # pcontour = np.append(np.array(feats[4]), vp.get_viewpoint('pitch_contour'), axis=0)
            #     # feats[4] = pcontour
            #     # dcontour = vp.gZZet_viewpoint('duration_contour')
            #     # # dcontour = np.append(np.array(feats[5]), vp.get_viewpoint('duration_contour'), axis=0)
            #     # feats[5] = dcontour
            #     # dratio = vp.get_viewpoint('duration_ratio')
            #     # # dratio = np.append(np.array(feats[6]), vp.get_viewpoint('duration_ratio'), axis=0)
            #     # feats[6] = dratio
            events, dists = vp.scale_sensitive_params()
            Features.append((events[0], events[1], dists[0], dists[1]))
    return Features




def _add_rests(chroma):
   arr = chroma.T.max(axis=1)
   rests = (arr < 0.15).astype('float64')
   c = np.append(chroma, np.array([rests]), axis=0)
   return c

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
    # files = librosa.util.find_files(copy_base)
    # fname = np.random.choice(files)

    # print(fname)
    # cut_file_1min_segments(fname)
    # signal, sr = librosa.load(fname)
    # three_mins = 3*get_one_min(sr)
    # one_min = get_one_min(sr)
    # signal = signal[-three_mins:-one_min]
    # raga = get_metadata_windows(fname)
    # print(raga)
    # pcd_X, pcd_Y = calculate_pcd(signal, sr)
    # plt.plot(pcd_X, pcd_Y, label=raga)
    # plt.show()

    # sample= (1, 2, 3)
    # X = np.zeros(5)
    # for i in range(5):
    #     X[i] = np.random.choice(sample)
    #
    # print(X)
    # print(_set_durations(X))
    # get_pcds_bigram([])
    chroma = np.zeros(20)
    for i in range(20):
        chroma[i] = int(np.random.choice((np.arange(12))))
    print(chroma)
    print(_calculate_dom_pitches(chroma).shape)