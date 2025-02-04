# Tacotron2-PyTorch
Yet another PyTorch implementation of [Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions](https://arxiv.org/pdf/1712.05884.pdf). The project is highly based on [these](#References). I made some modification to improve speed and performance of both training and inference.

## Requirements
- Python >= 3.8
- torch >= 1.0.0
- numpy
- scipy
- pillow
- inflect
- librosa
- Unidecode
- matplotlib
- tensorboardX
- resemblyzer
- tqdm

## Preprocessing
Support LJSpeech and LibriTTS.

## Training
1. For training Tacotron2, run the following command.

```bash
python3 train.py --data_dir=<dir/to/dataset> --preprocessed_data_dir=<dir/to/preprocessed> --ckpt_dir=<dir/to/models>
```

2. For training using a pretrained model, run the following command.

```bash
python3 train.py --data_dir=<dir/to/dataset> --preprocessed_data_dir=<dir/to/preprocessed> --ckpt_dir=<dir/to/models> --ckpt_pth=<pth/to/pretrained/model>
```

3. For using Tensorboard (optional), run the following command.

```bash
python3 train.py --data_dir=<dir/to/dataset> --preprocessed_data_dir=<dir/to/preprocessed> --ckpt_dir=<dir/to/models> --log_dir=<dir/to/logs>
```

You can find alinment images and synthesized audio clips during training. Recording freqency and text to synthesize can be set in `hparams.py`.

## Inference
- For synthesizing wav files, run the following command.

```bash
python3 inference.py --preprocessed_data_dir=<dir/to/preprocessed> --ckpt_pth=<pth/to/model> --img_pth=<pth/to/save/alignment> --wav_pth=<pth/to/save/wavs> --npy_pth=<pth/to/save/npys> --text=<text/to/synthesize>
```

## Pretrained Model
You can download pretrained models from [here](https://www.dropbox.com/sh/vk2erozpkoltao6/AABCk4WryQtrt4BYthIKzbK7a?dl=0) (git commit: [9e7c26d](https://github.com/BogiHsu/Tacotron2-PyTorch/commit/9e7c26d93ea9d93332b1c316ac85c58771197d4f)). The hyperparameter for training is also in the directory.

## Vocoder
Use Griffim-Lim or HifiGAN.

## Results
You can find some samples in [results](https://github.com/BogiHsu/Tacotron2-PyTorch/tree/master/results). These results are generated using either pseudo inverse or WaveNet.

The alignment of the attention is pretty well now (about 100k training steps), the following figure is one sample.

<img src="https://github.com/BogiHsu/Tacotron2-PyTorch/blob/master/results/tmp.png">

This figure shows the Mel spectrogram from the decoder without the postnet, the Mel spectrgram with the postnet, and the alignment of the attention.

## References
This project is highly based on the works below.
- [Tacotron2 by NVIDIA](https://github.com/NVIDIA/tacotron2)
- [Tacotron by r9y9](https://github.com/r9y9/tacotron_pytorch)
- [Tacotron by keithito](https://github.com/keithito/tacotron)
