import os
import argparse
from tqdm import tqdm


class LJSpeechPreprocessor(object):
    def __init__(self, raw_data_dir, preprocessed_data_dir):
        self.raw_data_dir = raw_data_dir
        self.preprocessed_data_dir = preprocessed_data_dir

    def preprocess(self):
        self.generate_metadata_txt()
        self.clean()
        self.train_val_test_split()
        self.log(f"Preprocess to {self.preprocessed_data_dir}, done.")

    def generate_metadata_txt(self):
        with open(f"{self.raw_data_dir}/metadata.csv", 'r', encoding='utf-8') as f:
            lines = []
            for line in tqdm(f):
                line = line.strip()
                if line[-1].isalpha():  # add missing periods
                    line += '.'
                lines.append(line)

        with open(f"{self.preprocessed_data_dir}/all.txt", 'w', encoding='utf-8') as f:
            for line in lines:
                f.write(line + '\n')
        self.log(f"Write metadata.csv to {self.preprocessed_data_dir}/all.txt.")

    def clean(self, *args, **kwargs):
        # TODO: Filter clean data. e.g. filter with length, filter noisy data  
        pass

    def train_val_test_split(self, *args, **kwargs):
        # TODO: Customize train/val/test split.
        lines = []
        with open(f"{self.preprocessed_data_dir}/all.txt", 'r', encoding='utf-8') as f:
            for line in f:
                lines.append(line)

        with open(f"{self.preprocessed_data_dir}/train.txt", 'w', encoding='utf-8') as f:
            for line in lines:
                f.write(line)
        self.log("Split done.")

    def log(self, msg):
        print("[LJSpeech Processor]: " + msg)


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    preprocessor = LJSpeechPreprocessor(args.data_dir, args.output_dir)
    preprocessor.preprocess()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # path
    parser.add_argument('-d', '--data_dir', type = str, required = True,
                        help = 'directory of raw data')
    parser.add_argument('-o', '--output_dir', type = str, required = True,
                        help = 'directory of processed data')

    args = parser.parse_args()
    main(args)
