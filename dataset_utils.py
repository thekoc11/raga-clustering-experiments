import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil
import librosa
import platform_details
from numba import vectorize
import audio_metadata
#
copy_base = platform_details.get_platform_path('Users/theko/Documents/Dataset')
ALL_COMPOSITIONS = [name for name in os.listdir(copy_base) if os.path.isdir(os.path.join(copy_base, name))]


# The following functions (`collate_raga_data`, `collate_and_add_celtic`, and `create_file_artist_genre_ids` ) are
# specific to the current dataset and must not be used
def collate_raga_data(base_dir, labels_json):
    IDX = os.listdir(base_dir)
    print(labels_json)
    LABELS = pd.read_json(labels_json, orient='index')
    AUDIO_DATA = pd.DataFrame()
    local_artists = {'': []}
    AUDIOS = []
    Audios_artist = []
    Audios_raga = []
    Audios_ragaId = []
    Audios_fileId = []
    genre_id = 0
    art_id = 0
    max_id_digits = 3
    for id in IDX:
        genre_id += 1
        raga_path = os.path.join(base_dir, id)
        local_artists[id] = os.listdir(raga_path)
        for artist in local_artists[id]:
            art_id += 1
            artist_path = os.path.join(raga_path, artist)
            i = 0
            for dirpth, dirname, fnames in os.walk(artist_path):
                for fname in [f for f in fnames if f.endswith('.mp3')]:
                    AUDIOS.append(os.path.join(dirpth, fname))
                    Audios_artist.append(artist)
                    Audios_raga.append(LABELS[0][id])
                    c_id = str(i + 1).zfill(max_id_digits)
                    Audios_ragaId.append(id)
                    Audios_fileId.append(c_id)
                    i = i + 1

    AUDIOS = pd.Series(AUDIOS)
    Audios_artist = pd.Series(Audios_artist)
    Audios_raga = pd.Series(Audios_raga)
    Audios_ragaId = pd.Series(Audios_ragaId)
    AUDIO_DATA['Path'] = AUDIOS
    AUDIO_DATA['Artist'] = Audios_artist
    AUDIO_DATA['Raga'] = Audios_raga
    AUDIO_DATA['ragaId'] = Audios_ragaId
    AUDIO_DATA['fileId'] = Audios_fileId

    return AUDIO_DATA


def collate_and_add_celtic(base_wes_dir, AUDIO_DATA):
    filt = AUDIO_DATA['Raga'] == 'Celtic'

    AUDIO_DATA = AUDIO_DATA[~filt]

    wes_iter = 0
    for dirpth, dirname, fnames in os.walk(base_wes_dir):
        for fname in [f for f in fnames if f.endswith('.mp3')]:
            AUDIO_DATA = AUDIO_DATA.append(
                {'Path': os.path.join(dirpth, fname), 'Artist': '', 'Raga': 'Celtic', 'ragaId': 'celtic',
                 'fileId': str(wes_iter + 1).zfill(3)}, ignore_index=True)
            wes_iter = wes_iter + 1

    print(AUDIO_DATA.loc[:, ['Path', 'fileId']])
    return AUDIO_DATA


def create_file_artist_genre_ids(AUDIO_DATA, copy=True):
    ar = AUDIO_DATA['Artist'].sort_values().unique()
    ra = AUDIO_DATA['Raga'].unique()

    ARTIST_ID = dict()
    RAGA_ID = dict()
    artist_id_pth = os.path.join(copy_base, 'v4_artist_ids.txt')
    genre_id_pth = os.path.join(copy_base, 'v4_genre_ids.txt')

    for i in range(len(ar)):
        ARTIST_ID[ar[i]] = i

    for i in range(len(ra)):
        RAGA_ID[ra[i]] = i + 1

    with open(artist_id_pth, 'w') as f:
        for key, val in (ARTIST_ID.items()):
            f.write(f"{key};{val}\n")
        f.close()

    with open(genre_id_pth, 'w', encoding="utf8") as f:
        f.write(f";0\n")
        f.write(f"unknown;0\n")
        for key, val in RAGA_ID.items():
            f.write(f"{key};{val}\n")
        f.close()

    AUDIO_DATA_copier = AUDIO_DATA.set_index('Path')
    for id in AUDIO_DATA_copier.index:
        AUDIO_DATA_copier.loc[id, 'fileId'] = (str(RAGA_ID[AUDIO_DATA_copier.loc[id, 'Raga']]).zfill(3) +
                                               str(ARTIST_ID[AUDIO_DATA_copier.loc[id, 'Artist']]).zfill(3) +
                                               AUDIO_DATA_copier.loc[id, 'fileId'])

    metadata_pth = os.path.join(copy_base, 'metadata.csv')

    with open(metadata_pth, 'w', encoding="utf8") as f:
        f.write(f"Path,Artist,Raga,RagaId,fileId\n")
        for path in AUDIO_DATA_copier.index:
            copy_dest = os.path.join(copy_base, AUDIO_DATA_copier.loc[path, 'fileId'])
            try:
                os.mkdir(copy_dest)
                if copy:
                    new_path = shutil.copy2(path, copy_dest)
                    new_path = new_path.strip(copy_base)
                    new_path = new_path.strip(r"/")
                    song_artist = AUDIO_DATA_copier.loc[path, 'Artist']
                    song_genre = AUDIO_DATA_copier.loc[path, 'Raga']
            except:
                pass

            f.write(
                f"{new_path},{song_artist},{song_genre},{[AUDIO_DATA_copier.loc[path, 'ragaId']]},{AUDIO_DATA_copier.loc[path, 'fileId']}\n")


def add_genre_to_dataset(genreID, genreFilesPath, copy=True):
    genre_id, genre = _create_Raga_ID_ditionary()
    files_list = librosa.util.find_files(platform_details.get_platform_path(genreFilesPath))
    artists = []
    fileIDs = []
    for i in range(len(files_list)):
        meta = audio_metadata.load(files_list[i])
        if meta['tags'].__contains__('artist'):
            artist = meta['tags'].artist
        else:
            artist = ''
        artists.append(artist)
        fileIDs.append(str(genreID).zfill(3) + str(0).zfill(3) + str(i+1).zfill(3))
    if genre.keys().__contains__(genreID):
        gen = genre[genreID]
    else:
        try:
            meta = audio_metadata.load(files_list[0])
            gen = meta['tags'].genre
        except:
            raise Exception("Add genre name to the 'v4_genre_ids' first")

    ### assigning artist ID := 0 for now, will edit the code later
    ### TODO: add get_artist() and get_artist_id()
    metadata_path = os.path.join(copy_base, 'metadata.csv')

    if copy:
        with open(metadata_path, 'a', encoding='utf8') as f:
            for i in range(len(files_list)):
                copy_dest = os.path.join(copy_base, fileIDs[i])
                try:
                    os.mkdir(copy_dest)
                except:
                    pass
                new_path = shutil.copy2(files_list[i], copy_dest)
                new_path = new_path.strip(copy_base)
                f.write(
                    f"{new_path},{artists[i]},{gen},{gen.lower()},{fileIDs[i]}\n")
                print(f"file {i + 1} of {len(files_list)} copied!!")



def get_random_songs(n=None, genres=(), artists=()):
    retVal = []
    if len(genres) == 0 and len(artists) == 0:
        n = len(ALL_COMPOSITIONS) if n is None else n
        retVal = np.random.choice(ALL_COMPOSITIONS, n)
    elif len(genres) != 0 and len(artists) == 0:
        Comps = []
        for g in genres:
            Comps.append(np.array(get_files_by_genre(g)))
        Comps = np.array(Comps)
        Comps = Comps.ravel()
        n = len(Comps) if n is None else n
        retVal = np.random.choice(Comps, n)
    ######
    # TODO: Add len(genres)== 0 and len(artists) != 0, len(genres) != 0 and len(artists) != 0
    ########

    return retVal


def get_metadata(filename):
    """
    Deprecated: Given the name of an audio file, this function returns the metadata (composer, genre and lyrics(if available))

    :param filename:
    :return:
    """
    import pandas as pd
    import os
    base_path = platform_details.get_platform_path(r"Users\theko\Documents\Dataset")
    metadata_path = os.path.join(base_path, 'metadata.csv')
    filename_new = filename.strip(base_path).replace("\\", "/")
    df = pd.read_csv(metadata_path, sep='\n')
    df[['Path', 'Artist', 'Raga', 'RagaId', 'fileId']] = df['Path,Artist,Raga,RagaId,fileId'].str.split(',',
                                                                                                        expand=True)
    df = df.drop(columns=['Path,Artist,Raga,RagaId,fileId'])
    df = df.set_index('Path')
    df = pd.DataFrame.drop_duplicates(df)
    # print(df)
    try:
        artist = df.loc[filename_new, 'Artist']
        genre = df.loc[filename_new, 'Raga']
        lyrics = ''
        return artist, genre, lyrics
    except:
        print(filename, filename_new, base_path)
        return 'unknown', 'Celtic', ''

# Private function that creates dictionaries : Raga -> RagaID, RagaID -> Raga and
# returns them
def _create_Raga_ID_ditionary():
    genre_ids = {}
    genres = {}
    genre_ids_file = platform_details.get_platform_path(r"Users\theko\Documents\Dataset\v4_genre_ids.txt")
    with open(genre_ids_file, 'r', encoding="utf-8") as f:
        for line in f:
            genre, genre_id = line.strip().split(';')
            genre_ids[genre] = int(genre_id)
            genres[int(genre_id)] = genre
    return genre_ids, genres

# Private function that creates dictionaries : Artist -> ArtistID, ArtistID -> Artist and
# returns them
def _creat_artist_ID_dictionary():
    artist_ids = {}
    artists = {}
    artist_ids_file = platform_details.get_platform_path(r"Users\theko\Documents\Dataset\v4_artist_ids.txt")
    with open(artist_ids_file, 'r', encoding="utf-8") as f:
        for line in f:
            artist, artist_id = line.strip().split(';')
            artist_ids[artist] = int(artist_id)
            artists[int(artist_id)] = artist
    return artist_ids, artists


def get_artist_ID(filename):
    """

    :param filename: conforms to the naming convention <genre_id><artist_id><file_id>
    :return: integer (artist ID)
    """
    name = np.array(list(filename), dtype='int32').reshape(3, 3)
    W = np.array([100, 10, 1])
    name = np.dot(name, W)
    return name[1]


def get_Artist(name):
    """

    :param name: conforms to the naming convention <genre_id><artist_id><file_id>
    :return: string (artist name)
    """
    artistId = get_artist_ID(name)
    artist_ids, artists = _creat_artist_ID_dictionary()
    if artists.__contains__(artistId):
        return artists[artistId]
    else:
        return None


def get_Raga_ID(filename):
    """

    :param filename: conforms to the naming convention <genre_id><artist_id><file_id>
    :return: integer (Raga ID)
    """
    name = np.array(list(filename), dtype='int32').reshape(3, 3)
    W = np.array([100, 10, 1])
    name = np.dot(name, W)
    return name[0]


def get_Raga_from_ID(ragaId):
    genre_ids, genres = _create_Raga_ID_ditionary()
    if genres.__contains__(ragaId):
        return genres[ragaId]
    else:
        return None


def get_Raga(name):
    ragaId = get_Raga_ID(name)
    genre_ids, genres = _create_Raga_ID_ditionary()
    if genres.__contains__(ragaId):
        return genres[ragaId]
    else:
        return None


def get_song_serial(filename):
    name = np.array(list(filename), dtype='int32').reshape(3, 3)
    W = np.array([100, 10, 1])
    name = np.dot(name, W)
    return name[2]


def get_files_by_genre(genreId):
    retFiles = []
    for f in ALL_COMPOSITIONS:
        if get_Raga_ID(f) == genreId:
            retFiles.append(f)
    return retFiles

def get_features_by_genre(genreId, feature='chroma'):
    path = platform_details.get_platform_path_custom('E', "Features/" + feature + "/stft/")
    all_features = os.listdir(path)
    Feats = []
    for f in all_features:
        if get_Raga_ID(f) == genreId:
            feat_path = os.path.join(path, f)
            feat = os.listdir(feat_path)[0]
            _feat = np.load(os.path.join(feat_path, feat))
            Feats.append(_feat)

    return Feats


def save_file_audio_data(song_name):
    path = os.path.join(copy_base, song_name)
    files = librosa.util.find_files(path, ext=['wav'])
    for f in files:
        y, sr = librosa.load(f)
        fname = f.replace('.wav', '.npy')
        np.save(fname, y, allow_pickle=False)


def load_audio_files(name, n, random=True, reverse=False):
    dir_path = os.path.join(copy_base, name)
    files = np.array(librosa.util.find_files(dir_path, ext=['wav']))
    if n == 'all':
        n = len(files)
        random = False

    if random:
        try:
            idx = np.random.choice(files.shape[0], n)
            return files[idx]
        except:
            print(f"failed: requested files - {n}, available files - {files.shape[0]}")
    elif reverse:
        try:
            files = files[::-1]  # syntax found on geeksforgeeks.org
            idx = np.arange(n)
            return files[idx]
        except:
            print(f"failed: requested files - {n}, available files - {files.shape[0]}")
    else:
        try:
            idx = np.arange(n)
            return files[idx]
        except:
            print(f"failed: requested files - {n}, available files - {files.shape[0]}")


if __name__ == '__main__':
    # AUDIOs = collate_raga_data('/mnt/c/RagaDataset/Carnatic/audio', '/mnt/c/RagaDataset/Carnatic/_info_/ragaId_to_ragaName_mapping.json')
    # m = AUDIOs.set_index('Path')
    # print(m.index)
    # AUDIOs = collate_and_add_celtic('/mnt/c/Dataset', AUDIOs)
    # create_file_artist_genre_ids(AUDIOs, copy=True)
    # add_genre_to_dataset(42, '/vismaya/mp3', copy=True)
    feats = get_features_by_genre(42)
    for f in feats:
        print(f.shape)
    print(len(feats))
