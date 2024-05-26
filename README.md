# DiffSal: Joint Audio and Video Learning for Diffusion Saliency Prediction
Official implementation of DiffSal, a diffusion-based generalized audio-visual saliency prediction framework using simple MSE objective function.

[![arXiv](https://img.shields.io/badge/ArXiv-2403.01226-orange)](https://arxiv.org/abs/2403.01226) [![projectpage](https://img.shields.io/badge/Project-Page-green)](https://junwenxiong.github.io/DiffSal/index.html)

 <!-- [![checkpoints](https://img.shields.io/badge/Model-Checkpoints-blue)]() -->



[Junwen Xiong](https://junwenxiong.github.io/), [Peng Zhang](https://teacher.nwpu.edu.cn/2011010119.html), Tao You, Chuanyue Li, Wei Huang, Yufei Zha

<br>
<img width="800" src="__asserts__/diffsal_2.png"/>
<br>


# 🔥 News
- **May. 26, 2024**. Training code released now! It's time to train DiffSal! 🚀🚀

# 📌 TODOs
- [ ] release pretrained weights.

# 🔧 Setup

### Environment Setup
Please install from ```requirements.txt``` .


```shell
conda create -n diff-sal python==3.10
conda activate diff-sal
pip install -r requirements.txt
```


# 🚅 How To Train 

🎉 To make our method reproducibl, we have released all of our training code. Four 4090 GPUs are enough :).

## Data Structure

The DiffSal model needs to be pre-trained on the DHF1k dataset.

```bash
./datasets/dhf1k
    ├── frames
    │   ├── 1
    │   │   ├── 1.png
    │   │   ├── 2.png
    │   │   ...
    │   │   ├── 100.png
    ├── maps
    │   ├── 1
    │   │   ├── 0001.png 
    │   │   ├── 0002.png
    │   │   ...
    │   │   ├── 0100.png
```
The DiffSal model is then fine-tuned on the audio-visual dataset.
```bash
./datasets/video_frames
    ├── AVAD
    │   ├── V1_Speech1
    │   │   ├── img_00001.jpg
    │   │   ├── img_00002.jpg
    │   │   ...
    │   │   ├── img_00100.jpg
./datasets/video_audio
    ├── AVAD
    │   ├── V1_Speech1
    │   │   ├── V1_Speech1.wav
./datasets/annotations
    ├── AVAD
    │   ├── V1_Speech1
    │   │   ├── maps
    │   │   │   ├── eyeMap_00001.jpg
./datasets/fold_lists/
    ├── AVAD_list_test_1_fps.txt
    ├── AVAD_list_test_2_fps.txt
    ├── AVAD_list_test_3_fps.txt
    ├── AVAD_list_train_1_fps.txt
    │    ...
```

## Running Scripts

The following is the pretrained training command:
```
sh scripts/train.sh
```
Then, use the following commands to fine-tune the model:
```
sh scripts/train_av.sh
```
If you want to infer the model, just remove ``--train``, set ``--test``, and leave the rest of the configuration unchanged.

# BibTeX

🌟 If you find our project useful in your research or application development, citing our paper would be the best support for us! 

```
@inproceedings{xiong2024diffsal,
    title={DiffSal: Joint Audio and Video Learning for Diffusion Saliency Prediction},
    author={Junwen Xiong, Peng Zhang, Tao You, Chuanyue Li, Wei Huang and Yufei Zha},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year={2024}
  }
```