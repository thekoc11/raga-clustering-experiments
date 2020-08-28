
import numpy as np
import matplotlib.pyplot as plt
import dataset_utils as du
import re

from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import os
import tqdm
from joblib import dump, load

y = np.load(os.path.join(du.copy_base,'DATASET_bin_Y.npy'))
X_Test = np.load(r"E:\DATASET\pcds_bigram\pcds_bigram_041.npy")

y_Test = np.zeros(len(X_Test), dtype='int32')
y_Test = y_Test + 22

Path = r"E:\DATASET"

# pattern = re.compile(r"^UniBiNoteDist_003_022_CFXthres49925eps\d+.npy$")
# pattern_test = re.compile(r"^pcds_UniBi_041_CFXeps\d+.npy$")
# pattern_test2 = re.compile(r"^pcds_UniBi_042_CFXeps\d+.npy$")


def build_patterns(ragaIds, features, chaos=False):
    ragas = ''
    for r in ragaIds:
        ragas = ragas + r + '_'

    if chaos:
        pattern = re.compile(rf"^{features}_{ragas}CFXthres\d+eps\d+.npy|^{features}_{ragas}CFXthres\d+eps\d+.joblib$")
    else:
        pattern = re.compile(rf"^{features}_{ragas.strip('_')}.npy|{features}_{ragas.strip('_')}.joblib$")
    return pattern

def get_inputs(path, ragaIds, features, chaos=False):
    files = os.listdir(path)
    matches = []
    # matches_TEST = []
    pattern = build_patterns(ragaIds, features, chaos)
    # pattern_test = build_patterns(ragaIds[-1:], features,chaos)
    for file in files:
        match = pattern.findall(file)
        # match_TEst = pattern_test.findall(file)
        if (len(match)) > 0:
            matches.append(match[0])
        # if (len(match_TEst)) > 0:
        #     matches_TEST.append(match_TEst[0])
    # print(len(matches_TEST))
    if len(matches) > 0:
        print(rf"{len(matches)} files found! ")
        return matches
    else:
        raise LookupError(rf"Ids {ragaIds} with features {features} not found at {path} pattern {pattern}")

    # Xs_Test = matches_TEST


def classify_svm_linear(Path, X_files, y, thresh=0.25, measures=('f1-score', 'precision')):
    """
    This  function runs the linear SVM classifier with numpy arrays whose names are passed in X_files as input data.
    :param Path: the path of the directory where the elements of :param X_files is present
    :param X_files: regex matches from get_inputs()
    :param y: labels
    :param thresh: If X is the output from ChaosFEX, we need to build a new classifier for every value of $\epsilon$
    :param measures: The array of measures over different files in X_files.
    :return: The array of accuracy metrics: for now, we are returning all (f1-score, precision and accuracy)
    """
    f1s = []
    precs = []
    accs = []
    models = []
    models_dir_path = os.path.join(r"E:\DATASET", 'models')
    try:
        os.mkdir(models_dir_path)
    except:
        pass
    for i in range(len(X_files)):
        x = np.load(os.path.join(Path, X_files[i]))
        # xii has been used to check whether the next x has the same shape as the current one.
        # Look further for implementation details
        xii = np.load(os.path.join(Path, X_files[(i+1) % len(X_files)]))
        # if len(x.shape) > 2, then flatten all individual values of x to make it a 2D array
        if len(x.shape) > 2:
            x_new = []
            xii_new = [] # this is not needed, but I just did this so that it was easier for me to understand what
            # was going on

            print(f"flattening the input shape {x.shape}, {xii.shape}")
            for ii in range(len(x)):
                # print(f"Before {x[i].shape}")
                x_new.append(x[ii].ravel())
                # print(f"After {x_new[i].shape}")
            for ij in range(len(xii)):
                xii_new.append(xii[ij].ravel())
            x = np.array(x_new, dtype='float64')
            xii = np.array(xii_new, dtype='float64')
            print(f"flatteining done! new shape: {x.shape}, {xii.shape}")
        # if shape of x is not equal to the shape of xii, then we assume this is a non-chaotic case, and
        # each raga which is to be the part of the training set is stored a separate numpy file.
        if x.shape[0] != xii.shape[0]:
            X = np.append(x, xii, axis=0)
            clf = svm.SVC(kernel='linear')
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True)
            print("10-fold cross-validation started")
            prec = np.zeros(10)
            f1 = np.zeros(10)
            acc = np.zeros(10)
            for iter in tqdm.tqdm(range(10)):
                clf.fit(X_train, y_train)
                predicted = clf.predict(X_test)
                if 'precision' in measures:
                    p = metrics.precision_score(y_test, predicted, pos_label=y[0])
                    prec[iter] = p
                if 'f1-score' in measures:
                    f = metrics.f1_score(y_test, predicted, pos_label=y[0])
                    f1[iter] = f
                if 'accuracy' in measures:
                    a = metrics.accuracy_score(y_test, predicted)
                    acc[iter] = a
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True)

            avg_prec = ''
            avg_f1 = ''
            avg_acc = ''
            if 'precision' in measures:
                avg_prec = prec.sum()/len(prec)
            if 'f1-score' in measures:
                avg_f1 = f1.sum() / len(f1)
            if 'accuracy' in measures:
                avg_acc = acc.sum()/len(acc)
            f1s.append(avg_f1)
            precs.append(avg_prec)
            accs.append(avg_acc)
            print(f"f1-score: {avg_f1}")

            models.append(clf)
            break
        else:
            if thresh == 0.499:
                X = x[:, :x.shape[1]//2]
            elif thresh == 0.25:
                X = x[:, x.shape[1]//2:]
            else:
                X = x
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True)

            prec = np.zeros(10)
            f1 = np.zeros(10)
            acc = np.zeros(10)
            clf = svm.SVC(kernel='linear')
            print(f"10-fold cross-validattion started for file {i+1} threshold: {thresh} ")
            for iter in tqdm.tqdm(range(10)):
                clf.fit(X_train, y_train)
                predicted = clf.predict(X_test)
                if 'precision' in measures:
                    p = metrics.precision_score(y_test, predicted, pos_label=y[0])
                    prec[iter] = p
                if 'f1-score' in measures:
                    f = metrics.f1_score(y_test, predicted, pos_label=y[0])
                    f1[iter] = f
                if 'accuracy' in measures:
                    a = metrics.accuracy_score(y_test, predicted)
                    acc[iter] = a
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True)
            avg_prec = ''
            avg_f1 = ''
            avg_acc = ''
            if 'precision' in measures:
                avg_prec = prec.sum() / len(prec)
            if 'f1-score' in measures:
                avg_f1 = f1.sum() / len(f1)
            if 'accuracy' in measures:
                avg_acc = acc.sum() / len(acc)
            f1s.append(avg_f1)
            precs.append(avg_prec)
            accs.append(avg_acc)

            print(f"f1-score: {avg_f1}")
            models.append(clf)

    f1s = np.array(f1s)
    precs = np.array(precs)
    accs = np.array(accs)
    ind = f1s.argmax()
    print(f"Max f1 {f1s.max()} achieved at iteration {ind}")
    model_name = os.path.join(models_dir_path, X_files[ind]).replace('.npy', '.joblib')
    print(f"saving the model to {model_name}")
    dump(models[ind], model_name)
    return f1s, precs, accs


if __name__ == '__main__':
    path = r"E:\DATASET\note_events"
    X_files = get_inputs(path, ('003', '022'), 'InvEvents')
    f1_1, prec_1, acc_1 = classify_svm_linear(path, X_files, y, thresh=2.0)