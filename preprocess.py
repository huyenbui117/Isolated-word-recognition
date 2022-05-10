import os
import sys
import argparse
from common import *
import shutil
from pydub import AudioSegment

class Segmenter:
    def __init__(self, start: float, end: float, label: str):
        self.start = start
        self.end = end
        self.label = label

    def __str__(self):
        return f"{self.start} - {self.end}: {self.label}"

    def __repr__(self):
        return self.__str__()

def preprocess(input_path, output_path, indicate = None):
    if (indicate is None or indicate == 0):
        for label in LABELS:
            os.mkdir(f"{output_path}/{label}")
    files = os.listdir(input_path)
    filenames = [file[:-4]for file in files]
    for file in filenames:
        labels = []
        with open(f"{input_path}/{file}.txt", "r") as f:
            for line in f:
                if (line.strip()==""):
                    continue
                start, end, label = line.split('\t')
                labels.append(Segmenter(float(start), float(end), label.strip()))
        for i, label in enumerate(labels):
            audio = AudioSegment.from_wav(f"{input_path}/{file}.wav")
            audio = audio[int(label.start * 1000) : int(label.end * 1000)]
            if indicate is None:
                audio.export(f"{output_path}/{label.label}/{int(file)}_{i+1}.wav",format= "wav")
            else:
                
                audio.export(f"{output_path}/{label.label}/{int(file)}_{i+1}_{indicate+1}.wav",format= "wav")
    print(f"{input_path} preprocessed! Saved in {output_path}")
def main():
    parser = argparse.ArgumentParser("")

    parser.add_argument(
        "--output_dir",
        nargs="?",
        default="data",
        help = "Path to processed data. "
    )

    parser.add_argument(
        "--overwrite", type=bool,
        nargs="?",
        default=False,
        help = "Overwrite the content of the preprocessed data directory. "
    )

    parser.add_argument(
        "--choose_data", type = int,choices=[0,1,2,3,4,5],
        nargs="?",
        default = 0,
        help = "Choose dataset to preprocess. 4 to preprocess all separatedly, 5 to combine all"
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
    if args.choose_data == 4:
        for path in DATA_PATH:
            _path = f"{args.output_dir}/{path}"
            os.mkdir(_path)
            
            preprocess(f"raw_data/{path}", _path)
    elif args.choose_data == 5:
        _path = f"{args.output_dir}/combined"
        os.mkdir(_path)
        for i, path in enumerate(DATA_PATH):
            preprocess(f"raw_data/{path}", _path, i)
    else:
        for i, path in enumerate(DATA_PATH):
            if args.choose_data == i:
                _path = f"{args.output_dir}/{path}"
                os.mkdir(_path)
                preprocess(f"raw_data/{path}", _path)

if __name__ == "__main__":
    main()
