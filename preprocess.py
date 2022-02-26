import os
import numpy as np
import argparse
from tqdm import tqdm
from resemblyzer import VoiceEncoder, preprocess_wav
import warnings
import json

from hparams import hparams as hps
from utils.audio import load_wav


warnings.simplefilter(action='ignore', category=FutureWarning)


class LJSpeechPreprocessor(object):
    def __init__(self, raw_data_dir, preprocessed_data_dir):
        self.raw_data_dir = raw_data_dir
        self.preprocessed_data_dir = preprocessed_data_dir

    def preprocess(self):
        self.generate_metadata_txt()
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

    def log(self, msg):
        print("[LJSpeech Processor]: " + msg)


class LibriTTSPreprocessor(object):
    def __init__(self, raw_data_dir, preprocessed_data_dir):
        self.raw_data_dir = raw_data_dir
        self.preprocessed_data_dir = preprocessed_data_dir

    def preprocess(self):
        self.dsets = {"train": "train-clean-100", "dev": "dev-clean", "test": "test-clean"}
        self.generate_metadata_txt()
        # self.extract_spk_dvectors()

    def generate_metadata_txt(self):
        for k, v in self.dsets.items():
            lines = []
            in_dir = f"{self.raw_data_dir}/{v}"
            for speaker in tqdm(os.listdir(in_dir), desc=k+"-speakers"):
                for chapter in os.listdir(os.path.join(in_dir, speaker)):
                    for file_name in os.listdir(os.path.join(in_dir, speaker, chapter)):
                        if file_name[-4:] != ".wav":
                            continue
                        base_name = file_name[:-4]
                        text_path = os.path.join(
                            in_dir, speaker, chapter, "{}.normalized.txt".format(base_name)
                        )
                        with open(text_path) as f:
                            text = f.readline().strip("\n")
                        wav_name = os.path.join(v, speaker, chapter, base_name)
                        lines.append(f"{wav_name}|{speaker}|{text}")

            # Filter length
            new_lines = []
            for line in tqdm(lines, desc="Filter out >15s audios."):
                wav_name = line.split("|")[0]
                wav = load_wav(f"{self.raw_data_dir}/{wav_name}.wav")
                if len(wav) > hps.sample_rate * 15:
                    print(f"{self.raw_data_dir}/{wav_name}.wav is too long.")
                else:
                    new_lines.append(line)
            lines = new_lines
            
            with open(f"{self.preprocessed_data_dir}/{k}.txt", 'w', encoding='utf-8') as f:
                for line in lines:
                    f.write(line + '\n')
            self.log(f"Write to {self.preprocessed_data_dir}/{k}.txt.")

    def extract_spk_dvectors(self):
        spk_wavs = {}
        for k in self.dsets:
            with open(f"{self.preprocessed_data_dir}/{k}.txt", 'r', encoding='utf-8') as f:
                for line in f:
                    if line == "\n":
                        continue
                    wav_name, spk, _ = line.strip().split("|")
                    if spk not in spk_wavs:
                        spk_wavs[spk] = [wav_name]
                    else:
                        spk_wavs[spk].append(wav_name)

        # Dvector extraction.
        encoder = VoiceEncoder()
        spk_embeds = {}
        for spk, wav_names in spk_wavs.items():
            spk_embeds[spk] = []
            for wav_name in tqdm(wav_names, desc=f"Extract dvectors (speaker {spk})"):
                wav = preprocess_wav(f"{self.raw_data_dir}/{wav_name}.wav")
                embed = encoder.embed_utterance(wav)
                spk_embeds[spk].append(embed)

        # Average over all utterances and save.
        os.makedirs(f"{self.preprocessed_data_dir}/dvector", exist_ok=True)
        for spk in spk_embeds:
            avg_embed = np.mean(np.stack(spk_embeds[spk]), axis=0)
            with open(f"{self.preprocessed_data_dir}/dvector/{spk}.npy", 'wb') as f:
                np.save(f, avg_embed)
        with open(f"{self.preprocessed_data_dir}/speakers.json", 'w', encoding="utf-8") as f:
            speakers = list(spk_embeds.keys())
            speakers = {spk: i for i, spk in enumerate(speakers)}
            json.dump(speakers, f, indent=4)
        self.log(f"Write dvectors to {self.preprocessed_data_dir}/dvector.")

    def log(self, msg):
        print("[LibriTTS Processor]: " + msg)


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    # preprocessor = LJSpeechPreprocessor(args.data_dir, args.output_dir)
    preprocessor = LibriTTSPreprocessor(args.data_dir, args.output_dir)
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
