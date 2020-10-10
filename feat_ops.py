import numpy as np
from joblib import Parallel, delayed
import dataset_utils
import audio_utils
import platform_details

Merged_feats = []

def _merger(feat_set):
    for f in feat_set:
        for t in f:
            Merged_feats.append(t)

def get_merged_genre_feats(genreId, feature='chroma'):
    feats = dataset_utils.get_features_by_genre(genreId, feature)
    Merged_feats = []
    for feat in feats:
        for f in feat:
            Merged_feats.append(f)
    return np.array(Merged_feats)


if __name__ == '__main__':
    print(get_merged_genre_feats(41).shape)