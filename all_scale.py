from itertools import chain

import numpy as np
from sklearn.model_selection._split import _validate_shuffle_split, StratifiedShuffleSplit
from sklearn.utils import indexable, _safe_indexing
from sklearn.utils.validation import _num_samples

from Viewpoints import Viewpoints

def get_data_aug(data):
    """
    Takes data in a certain scale as argument, and returns the same array in all different scales
    :param data: An array of pitches + rests. The zeroth axis should always contain 13 elements(12 pitches + 1 rest)
        shape: (13, ) or (13, *)
    :return: The input array in all scales
        shape: (12, 13, ) or (12, 13, *)
    """
    if data.shape[0] != 13:
        raise TypeError(f"data.shape[0] is {data.shape[0]} while the expected value is 13")

    aug_data_shape = [12] + list(data.shape)
    aug_data = np.zeros(aug_data_shape)
    sub_data = data[:12]
    for i  in range(12):
        arr = np.roll(sub_data, i, axis=0)
        aug_data[i] = np.append(arr, data[12:], axis=0)

    return aug_data

def get_one_hot_rep(data):
    """
    In case the data is not an array of shape (13, *) (for example, if it is the most prominent pitches at every
    time stamp), but contains integers in range [0, 12] as values, then we can convert it into a 2D array of shape (13, *).
    e.g. data = [0, 1, 0, 2, 2, 1, 1, 2, 2, 0],
    then, the output from this function will be:
    out = [[1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],(highlights all positions of 0)
           [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],(highlights all positions of 1)
           [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0]](highlights all positions of 2)

    :param data:
    :return:
    """
    if data.dtype == 'int32':
        one_hot_data = np.zeros((13, data.shape[0]))
        for j in range(len(data)):
            one_hot_data[data[j]][j] = 1
        data = one_hot_data
        return data
    else:
        raise TypeError('Data of type int32 is expected')

def get_linear_rep(data, data_shape):
    """
    This function converts the output from get_data_aug() back to normal, i.e.
    if data was converted to a one-hot representation using get_one_hot_rep()
    before passing it to get_data_aug(), then this is an inverse function of
    get_one_hot_rep()
    :param data: the output from get_data_aug()
    :param data_shape: original data shape (before get_one_hot_rep())
    :return: augmented data in original shape
    """
    if len(data_shape) == 1:
        retVal = np.zeros((data.shape[0], data_shape[0]))
        for i in range(len(data)):
            retVal[i] =data[i].argmax(axis=0)
    else:
        retVal = data

    return retVal



class Augmentor:
    def __init__(self, data, type='dists', axis=0):
        self.X = data
        self.data_shape = data.shape
        self.type = type
        # As of now, there are only 4 scale-sensitive features: pcds(unigram & bigram), pitches and chromagrams.
        # any n-gram pcds are perceived as distributions, while pitches and chromagrams are perceived as events.
        if type == 'dists':
            if data.shape[-1]%13 != 0:
                raise AssertionError(f"data with shape {data.shape} may not necessarily be a distribution over pitches")
        elif type == 'events':
            if 13 in data.shape: # if this is true, data this is a chromagram
                ax_ind = data.shape.index(13)
                self.X = data.reshape(data.shape[ax_ind], -1)
            elif (max(data) in range(13)) and min(data) == 0.0: # if this is true, then data is an array of pitch events
                data = data.astype('int32')
                self.X = get_one_hot_rep(data)

    def get_augmented_data(self):
        aug_data = get_data_aug(self.X)
        if self.type == 'events': # if data was pitch events, then convert them back to the original form
            aug_data = get_linear_rep(aug_data, self.data_shape)
        return aug_data


def train_test_split(*arrays, test_size=None, train_size=None, shuffle=True, random_split=None, type='dists'):
    """
    This function is a modification of sklearn.model_selection.train_test_split(). Here, we split the data into
    into train and test sets, and then augment the train data
    :param arrays:
    :param test_size:
    :param train_size:
    :param shuffle:
    :param random_split:
    :param type: The type of data that the Augmentor shall perceive
    :return:
    """
    n_arrays = len(arrays)
    if n_arrays == 0:
        raise ValueError("At least one array required as input")

    arrays = indexable(*arrays)
    n_samples = _num_samples(arrays[0])
    # print(f"n_samples: {n_samples}")
    n_train, n_test = _validate_shuffle_split(n_samples, test_size, train_size,
                                              default_test_size=0.1)
    # print(n_train, n_test)
    if shuffle is False:
        train = np.arange(n_train)
        test = np.arange(n_train, n_train + n_test)
    else:
        cv = StratifiedShuffleSplit(test_size=n_test, train_size=n_train, random_state=random_split)
        if len(arrays) >= 2:
            y = arrays[1]
        else:
            y = None
        train, test = next(cv.split(X=arrays[0], y=y))

    # print(f"trains: {train}, test: {test}")
    List =  list(chain.from_iterable((_safe_indexing(a, train),
                                     _safe_indexing(a, test)) for a in arrays))

    ###############################################################################
    # What follows is BAD CODE: I'm assuming that the inputs arrays (*arrays)
    # will always be of the form X, y, where X is the training data and y are the
    # labels.
    ###############################################################################

    X_train = List[0]
    y_train = List[2]
    X_train_final = []
    y_train_final = []
    for i in range(len(X_train)):
        X_train_augmentor = Augmentor(X_train[i], type)
        X_train_augmented = X_train_augmentor.get_augmented_data()
        # print(f"the datatype of y:{y_train.dtype}, y itself: {y_train[i]} ")
        # print(X_train_augmented.shape)
        y = np.ones(12) * y_train[i]
        X_train_final = X_train_final + [x for x in X_train_augmented]
        y_train_final = y_train_final + [y_i for y_i in y]

    X_train_final = np.array(X_train_final)
    y_train_final = np.array(y_train_final)

    return X_train_final, List[1], y_train_final, List[-1]



def main():
    chroma0 = np.random.normal(0, 1, (46, 13, 2585))
    chroma1 = np.random.normal(0, 1, (50, 13, 2585))
    feats0 = []
    feats1 = []
    y0 = []
    y1 = []

    for c in chroma0:
        vp = Viewpoints(c)
        events, dists = vp.scale_sensitive_params()
        # print(events[0].shape, events[1].shape)
        feats0.append(events[1])
        y0.append(3)
    for c in chroma1:
        vp = Viewpoints(c)
        events, dists = vp.scale_sensitive_params()
        # print(events[0].shape, events[1].shape)
        feats1.append(events[1])
        y1.append(42)
    X = np.array(feats0 + feats1)
    y = np.array(y0 + y1)
    # print(X.shape, y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, type='events')

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)



if __name__ == '__main__':
    main()





