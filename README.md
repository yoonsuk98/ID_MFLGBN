# Multiscale Fusion Based Local-Global Bridging Network for Lightweight Image Denosing
[Yoonsuk Kang](https://github.com/yoonsuk98), [The Van Le](https://github.com/vvvanthe), [Jin Young Lee](https://sites.google.com/view/ivcl/research?authuser=0/)

Intelligent Visual Computing Laboratory (IVCL), South Korea

---

This repository is the official PyTorch implementation of [Multiscale Fusion Based Local-Global Bridging Network for Lightweight Image Denosing]()


> Image denoising is the process of restoring images contaminated by internal random noise from electronic devices. Recent image denoising methods utilize deep learning networks to effectively remove the noise. However, many state-of-the-art denoising networks face challenges in balancing the preservation of fine local details with the demand to capture global contextual information. In addition, they often find it difficult to maintain a good trade-off between computational complexity and denoising performance. In order to address these challenges, we propose a multiscale fusion based local-global bridging network (MFLGBN) that effectively learns full-range dependencies while maintaining low computational complexity. Particularly, MFLGBN consists of a multiscale fusion block (MFB) and a local-global bridging block (LGBB). MFB explicitly learns mid-range dependencies by fusing features via grouped convolutions (GConv), whereas LGBB bridges CNN- and transformer-based attention modules to unify local and global feature extraction, enabling the capture of both short- and long-range dependencies. These blocks are arranged sequentially to model full-range dependencies, which enables the recovery of fine-grained local textures with global information.

<p align=center><img width="80%" src="figs\fig1.png"/></p>



### Quick start


1. [Install CUDA](https://developer.nvidia.com/cuda-downloads)

2. [Install PyTorch](https://pytorch.org/get-started/locally/)

3. Install dependencies
```bash
pip install -r requirements.txt
```

### Dataset Preparation

Training and testing sets can be downloaded as follows. Please put them in `trainsets`, `testsets` respectively.


#### Gaussian noise

  - Training: [DIV2K](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar) (800 training images) + [Flickr2K](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) (2650 images) + [BSD500](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz) (400 training&testing images) + [WED](http://ivc.uwaterloo.ca/database/WaterlooExploration/exploration_database_and_code.rar)(4744 images)
  - Testing: CBSD68 + Kodak24 + McMaster + Urban100 [(download all)](https://github.com/cszn/FFDNet/tree/master/testsets)

#### Real-noise images

  - Training : [SIDD](https://github.com/swz30/Restormer/blob/main/Denoising/README.md#training-1)
  - Testing : SIDD + DND [download all](https://github.com/swz30/Restormer/blob/main/Denoising/README.md#training-1)
  

### Training

#### Gaussian-noise 
run the following commands for Gaussian-noise training. You may need to change the configurations in the json file for different settings, such as number of gpu, training path, etc.

```bash
python TRAIN_ORG_PSNR.py --opt ./options/denoise/train_PROPOSED_denoise_15.json # can change 15 -> 25 or 50

"""
--opt: json file path.

"""
```

#### Real-noise 
run the following commands for Real-noise training. You may need to change the configurations in the json file for different settings, such as number of gpu, training path, etc.

```bash
python TRAIN_ORG_PSNR.py --opt ./options/denoise_realworld/train_PROPOSED_realworld.json

"""
--opt: json file path.

"""
```

### Make Test Files (only for Real-noise)
Following command to make test files in Real noise.


```bash
# SIDD
python TEST_REALWORLD_SIDD.py --opt ./options/denoise_realworld/train_PROPOSED_denoise_realworld.json --model_name proposed

# DND
python TEST_REALWORLD_DND.py --opt ./options/denoise_realworld/train_PROPOSED_denoise_realworld.json --model_name proposed


"""
--model_name: saved parameter model names

"""
```

### Testing

#### Gaussian-noise 
Following command to run testing in Gaussian noise.

```bash
python TEST_GAUSSIAN.py --opt ./options/denoise/train_PROPOSED_denoise_15.json --test_set CBSD68 --noise 15

"""
--test set: test set names (CBSD68, Kodak24, McMaster, urban100)
--noise: intensity of gaussian noise(15, 25, 50)

"""

```

#### Real-noise 
Following command to run testing in Real noise.

```bash
# SIDD (execute in matlab)
evaluate_sidd.m

# DND 
submit to website: https://noise.visinf.tu-darmstadt.de/submit/
```
---

## Citation
    @article{,
    author={Yoonsuk Kang, The Van Le and Jin Young Lee},
    journal={IEEE Transactions on Multimedia}, 
    title={Multiscale Fusion Based Local-Global Bridging Network for Lightweight Image Denosing}, 
    year={2025},
    volume={},
    number={},
    pages={},
    keywords={Lightweight Image denoising; Multiscale Fusion},
    doi={}}