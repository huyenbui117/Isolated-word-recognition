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
def classification(sample, templates):
    distances = dict()
    for label in LABELS:
        distances[label] = []
        for template in templates[label]:
            distances[label].append(dtw(sample[0], template[0]))

    # find the shortest distance for each label
    min_distances = {}
    for label in LABELS:
        min_distances[label] = min(distances[label])

    # print(min_distances)

    # shortest distance is the match
    min_label = min(min_distances, key=min_distances.get)

    return min_label
def main():
    LABELS.remove('sil')
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path_to_feature", type=str,
        default= "feature",
        nargs="?", 
        help= "Path to feature directory, containing *.sav files"
    )
    parser.add_argument(
        "path_to_template", type=str,
        default= "templates.tpl",
        nargs="?", 
        help= "Path to template file *.tpl"
    )
    args = parser.parse_args()
    samples = load_features(args.path_to_feature)
    templates = pickle.load(open(args.path_to_template,"rb"))
    for label in LABELS:
        correct_label = label
        predicted_label = [classification(sample, templates)for sample in samples[label]]
        
        accuracy = sum([1 for label in predicted_label if label == correct_label]) / len(predicted_label) * 100
        distribution = {}
        for label in LABELS:
            distribution[label] = predicted_label.count(label) / len(predicted_label) * 100

        # sort distribution
        sorted_distribution = sorted(distribution.items(), key=lambda x: x[1])
        sorted_distribution.reverse()

        print(f"Label {correct_label}, Accuracy: {accuracy:.2f}%", end=", ")

        for label, percentage in sorted_distribution:
            print(f"{label}: {percentage:.2f}%", end=", ")

        print()
if __name__ == '__main__':
    main()