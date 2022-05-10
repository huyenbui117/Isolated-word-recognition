import random
import os
import shutil
import argparse
import pickle
from common import *
from feature_extraction import load_features
def main():
    LABELS.remove("sil")
    try:
        shutil.rmtree("sample_data")
    except FileNotFoundError:
        pass

    os.mkdir("sample_data")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path_to_feature", type=str,
        default= "feature/combined",
        nargs="?", 
        help= "Path to feature directory, containing *.sav files"
    )
    args = parser.parse_args()
    features = load_features(args.path_to_feature)
    templates = dict()
    for label in LABELS:
        _template = random.sample(features[label], SAMPLE_COUNT)
        templates[label] = _template
    pickle.dump(templates, open(f"templates.tpl", "wb"))
if __name__ == '__main__':
    main()