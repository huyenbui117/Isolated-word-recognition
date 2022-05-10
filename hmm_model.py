import numpy as np
from feature_extraction import load_features
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from hmmlearn import hmm
import argparse
import pickle
from common import *
def data_preparation():
    idx_labels = dict()
    data = load_features("feature")
    for label in LABELS:
        for i,d in enumerate(data[label]):
            data[label][i]=np.array(data[label][i][0]).T
    for label in LABELS:
        idx_labels[label] = [LABELS.index(label)] * len(data[label])
    X = {'train': {}, 'test': {}}
    y = {'train': {}, 'test': {}}

    for label in LABELS:
        x_train, x_test, y_train, y_test = train_test_split(data[label], idx_labels[label], test_size=0.2)

        X['train'][label] = x_train
        X['test'][label] = x_test
        y['train'][label] = y_train
        y['test'][label] = y_test
    # for label in LABELS:
    #     print(f"{label}: {len(X['train'][label])} / {len(X['test'][label])}")
    # print(X['train']['A'][0].shape)

    return X, y
def train(X, y):
    models= dict()
    for i, label in enumerate(LABELS):
        models[label] = hmm.GMMHMM(n_components=STATES[i], covariance_type="diag", n_iter=300)
        models[label].fit(X=np.vstack(X['train'][label]), lengths=[x.shape[0] for x in X['train'][label]])
    save_model(models)
    return models
def evaluate(X, y, models):
    y_true = []
    y_preds = []

    for label in LABELS:
        for mfcc, target in zip(X['test'][label], y['test'][label]):
            scores = [models[label].score(mfcc) for label in LABELS]
            preds = np.argmax(scores)
            y_true.append(target)
            y_preds.append(preds)
    return y_true, y_preds
def save_model(models):
    pickle.dump(models,open("hmm_model.pkl","wb"))
def load_model():
    return pickle.load(open("hmm_model.pkl","rb"))
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--do_train", type=bool,
        default= True,
        nargs="?",
        help="Set to True to train the model, if false load hmm_model.pkl instead"
    )
    parser.add_argument(
        "--do_evaluate", type=bool,
        default=True,
        nargs="?",
        help="Set to True to evaluate the model"
    )
    args = parser.parse_args()
    LABELS.remove('sil')
    X, y = data_preparation()
    if args.do_train:
        models = train(X, y)
    else:
        models=load_model()
    y_true, y_preds = evaluate(X, y, models)
    report = classification_report(y_true, y_preds)
    print(report)
if __name__ == '__main__':
    main()