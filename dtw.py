import librosa
import IPython
import pickle
import argparse
import os
from feature_extraction import load_features
from common import *
def dtw(feature, reference_feature):
    D, wp = librosa.sequence.dtw(feature, reference_feature, subseq=True, metric="euclidean")
    return D[-1, -1] / wp.shape[0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path_to_feature", type=str,
        default= "feature/combined",
        nargs="?", 
        help= "Path to feature directory, containing *.sav files"
    )
    args = parser.parse_args()
    features = load_features(args.path_to_feature)
    
    IPython.embed()
    pass
if __name__ == '__main__':
    main()