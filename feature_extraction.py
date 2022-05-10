import librosa
import numpy as np
import os
import argparse
import shutil
import pickle
from common import *
import IPython
def mfcc_extraction(file_path:str):
    sound, sr = librosa.load(file_path)
    mfcc = librosa.feature.mfcc(
        y = sound, n_mfcc = 13, sr = sr, n_mels=128, fmax=8000, power=2,n_fft=1024
    )
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order = 2)
    return np.concatenate((mfcc, delta_mfcc, delta2_mfcc))
def feature_extraction(data_dir:str, output_path):
    
    for label in LABELS:
        path = f"{data_dir}/{label}"
        files = os.listdir(path)
        features=[]
        for file in files:
            # print(f"{file}, {label}")
            feature = mfcc_extraction(f"{path}/{file}")
            features.append([feature,file[:-4]])
        pickle.dump(features, open(f"{output_path}/{label}.sav", "wb"))
    print(f"Extracted features from {data_dir}. Saved in {path}.sav")
def load_features(path_to_feature:str):
    files = os.listdir(path_to_feature)
    features = dict()
    for file in files:
        _path = f"{path_to_feature}/{file}"
        features[f"{file[:-4]}"]=pickle.load(open(_path,"rb"))
    return features       
def main():
    LABELS.remove("sil")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str,
        nargs = "?",
        default = "data/19020063_CaoDinhHoangMinh",
        help="Path to dataset"
    )
    parser.add_argument(
        "--multiple_dataset", type=bool,
        nargs = "?",
        default = False,
        help =  "Set to true to extract multiple dataset in the same data_dir"
    )
    parser.add_argument(
        "output_dir", type=str,
        nargs = "?",
        default = "feature",
        help = "Path to folder containing extracted features"
    )
    parser.add_argument(
        "--overwrite", type=bool,
        nargs="?",
        default=False,
        help = "Overwrite the content of the extracted features directory. "
    )

    args = parser.parse_args()
    if (os.path.exists(args.output_dir)):
        if not args.overwrite:
            raise ValueError(
                f"Output directory ({args.output_dir}) is already existed and is not empty. Use --overwrite to overcome. "
            )
        else:
            shutil.rmtree(args.output_dir)
    os.mkdir(args.output_dir)

    if args.multiple_dataset == True:
        datasets = os.listdir(args.data_dir)
        for dataset in datasets:
            input_path = f"{args.data_dir}/{dataset}"
            output_path = f"{args.output_dir}/{dataset}"
            if not os.path.exists(output_path):
                os.mkdir(output_path)
            feature_extraction(input_path, output_path)
    else:
        feature_extraction(args.data_dir, args.output_dir)
        
if __name__ == '__main__':
    main()