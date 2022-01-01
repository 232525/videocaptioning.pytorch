# pytorch implementation of video captioning

recommend installing pytorch and python packages using Anaconda

This code is based on [video-caption.pytorch](https://github.com/xiadingZ/video-caption.pytorch)

## requirements (my environment, other versions of pytorch and torchvision should also support this code (not been verified!))

- cuda
- pytorch 1.7.1
- torchvision 0.8.2
- python 3
- ffmpeg (can install using anaconda)

### python packages

- tqdm
- pillow
- nltk

## Data

[MSR-VTT](https://www.mediafire.com/folder/h14iarbs62e7p/shared). Download and put them in `./data/msr-vtt-data` directory

```bash
|-data
  |-msr-vtt-data
    |-train-video
    |-test-video
    |-annotations
      |-train_val_videodatainfo.json
      |-test_videodatainfo.json
```

[MSVD](https://www.cs.utexas.edu/users/ml/clamp/videoDescription/). Download and put them in `./data/msvd-data` directory

```bash
|-data
  |-msvd-data
    |-YouTubeClips
    |-annotations
      |-AllVideoDescriptions.txt
```

## Options

all default options are defined in opt.py or corresponding code file, change them for your like.

## Acknowledgements
Some code refers to [ImageCaptioning.pytorch](https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning)

## Usage

### (Optional) c3d features
you can use [video-classification-3d-cnn-pytorch](https://github.com/kenshohara/video-classification-3d-cnn-pytorch) to extract features from video. 

### Steps
0. preprocess MSVD annotations (convert txt file to json file)

refer to `data/msvd-data/annotations/prepro_annotations.ipynb`

1. preprocess videos and labels

```bash
# For MSR-VTT dataset
CUDA_VISIBLE_DEVICES=0 python prepro_feats.py --video_path ./data/msr-vtt-data/train-video --output_dir ./data/msr-vtt-data/resnet152 --model resnet152 --n_frame_steps 40

CUDA_VISIBLE_DEVICES=0 python prepro_feats.py --video_path ./data/msr-vtt-data/test-video --output_dir ./data/msr-vtt-data/resnet152 --model resnet152 --n_frame_steps 40

python prepro_vocab.py --input_json data/msr-vtt-data/annotations/train_val_videodatainfo.json data/msr-vtt-data/annotations/test_videodatainfo.json --info_json data/msr-vtt-data/info.json --caption_json data/msr-vtt-data/caption.json --word_count_threshold 4

# For MSVD dataset
CUDA_VISIBLE_DEVICES=0 python prepro_feats.py --video_path ./data/msvd-data/YouTubeClips --output_dir ./data/msvd-data/resnet152 --model resnet152 --n_frame_steps 40

python prepro_vocab.py --input_json data/msvd-data/annotations/MSVD_annotations.json --info_json data/msvd-data/info.json --caption_json data/msvd-data/caption.json --word_count_threshold 2
```

2. Training a model

```bash
# For MSR-VTT dataset
CUDA_VISIBLE_DEVICES=0 python train.py \
    --epochs 1000 \
    --batch_size 300 \
    --checkpoint_path data/msr-vtt-data/save \
    --input_json data/msr-vtt-data/annotations/train_val_videodatainfo.json \
    --info_json data/msr-vtt-data/info.json \
    --caption_json data/msr-vtt-data/caption.json \
    --feats_dir data/msr-vtt-data/resnet152 \
    --model S2VTAttModel \
    --with_c3d 0 \
    --dim_vid 2048

# For MSVD dataset
CUDA_VISIBLE_DEVICES=0 python train.py \
    --epochs 1000 \
    --batch_size 300 \
    --checkpoint_path data/msvd-data/save \
    --input_json data/msvd-data/annotations/train_val_videodatainfo.json \
    --info_json data/msvd-data/info.json \
    --caption_json data/msvd-data/caption.json \
    --feats_dir data/msvd-data/resnet152 \
    --model S2VTAttModel \
    --with_c3d 0 \
    --dim_vid 2048
```

3. test

    opt_info.json will be in same directory as saved model.

```bash
# For MSR-VTT dataset
CUDA_VISIBLE_DEVICES=0 python eval.py \
    --input_json data/msr-vtt-data/annotations/test_videodatainfo.json \
    --recover_opt data/msr-vtt-data/save/opt_info.json \
    --saved_model data/msr-vtt-data/save/model_xxx.pth \
    --batch_size 100

# For MSVD dataset
CUDA_VISIBLE_DEVICES=0 python eval.py \
    --input_json data/msvd-data/annotations/test_videodatainfo.json \
    --recover_opt data/msvd-data/save/opt_info.json \
    --saved_model data/msvd-data/save/model_xxx.pth \
    --batch_size 100
```

## NOTE
This code is just a simple implementation of video captioning


## Acknowledgements
Some code refers to [ImageCaptioning.pytorch](https://github.com/ruotianluo/ImageCaptioning.pytorch)
