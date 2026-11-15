<div align="center">

<p align="center" style="border-radius: 10px">
  <img src="asset/logo3.png" width="35%" alt="logo"/>
</p>

<h1>MobileI2V: Fast and High-Resolution Image-to-Video on Mobile Devices</h1>


[Shuai Zhang](https://github.com/Shuaizhang7)<sup>\*</sup>, [Bao Tang](https://github.com/Tt-DAY)<sup>\*</sup>, [Siyuan Yu](https://github.com/siyryu)<sup>\*</sup>, [Yueting Zhu](https://github.com/lazypomeloo), [Jingfeng Yao](https://github.com/JingfengYao),<br>[Ya Zou](https://github.com/ZouYa99), [Shanglin Yuan](https://github.com/ysl6323), [Li Yu](https://eic.hust.edu.cn/professor/yuli/), [Wenyu Liu](http://eic.hust.edu.cn/professor/liuwenyu), [Xinggang Wang](https://xwcv.github.io/index.htm)<sup>📧</sup>


Huazhong University of Science and Technology (HUST) 

(\* equal contribution, 📧 corresponding author)

[![Project](https://img.shields.io/static/v1?label=Project&message=Github&color=blue&logo=github-pages)](https://hustvl.github.io/MobileI2V/)
[![arxiv paper](https://img.shields.io/badge/arXiv-Paper-red)](https://arxiv.org/abs/2511.21475)
[![checkpoints](https://img.shields.io/badge/HuggingFace-🤗-green)](https://huggingface.co/hustvl/MobileI2V)

</div>

## 📰 News
- **[2026.05.20]** We open-sourced our [distillation code](https://github.com/hustvl/MobileI2V/tree/distillation).
- **[2025.11.27]** We have released our paper on [arXiv](https://arxiv.org/abs/2511.21475) and our base model code.

## 📄 Introduction
<div align="center">
<img src="./asset/fig1.png">
</div>
Compared with SVD-XT (1.5B), our 5.55× smaller MobileI2V (0.27B) achieves similar generation quality, using only 2.24s on mobile and running 199× faster on an A100 GPU.

## 🎯 Demo

#### (1) 1280×720×17 Image to Video
<div align="center">
  <img src="./asset/videos/video1.gif" width="24.5%">
  <img src="./asset/videos/video2.gif" width="24.5%">
  <img src="./asset/videos/video3.gif" width="24.5%">
  <img src="./asset/videos/video4.gif" width="24.5%">
</div>

#### (2) 960×960×17 Image to Video
<div align="center">
  <img src="./asset/videos/video5.gif" width="18.5%">
  <img src="./asset/videos/video6.gif" width="18.5%">
  <img src="./asset/videos/video7.gif" width="18.5%">
  <img src="./asset/videos/video8.gif" width="18.5%">

</div>



## 🎯 How to Use

### Installation
You can install the required environment using the provided requirements.txt file. 

```
pip install -r requirements.txt
```
### Data Processing
There are many open source video datasets, such as [Openvid](https://github.com/NJU-PCALab/OpenVid-1M), [VFHQ](https://liangbinxie.github.io/projects/vfhq/) and [Celebv-text](https://github.com/CelebV-Text/CelebV-Text). The video should be cut into a fixed number of frames (such as 17 or 25...), and the video data should be filtered based on aesthetic (use [DOVER](https://github.com/VQAssessment/DOVER)) and optical flow scores (refer to OpenSora [data Processing](./tools/scoring/README.md)).


You should organize your processed train data into a CSV file, as shown below:

```
video_path,text,num_frames,height,width,flow
./_JnC_Zj_P7s_22_0to190_extracted.mp4,scenery,17,720,1080,3.529723644
./_JnC_Zj_P7s_22_0to190_extracted.mp4,scenery,17,720,1080,4.014187813
```

### Train
You can use the provided ./train_scripts/train_i2v.sh script for training. The configuration file is located at: ./configs/mobilei2v_config/. Before training, download the weights for [video-vae](https://huggingface.co/Lightricks/LTX-Video/tree/main/vae) and [qwen2-0.5B](https://huggingface.co/Qwen/Qwen2-0.5B/tree/main) and replace the model path in the configuration file.
```
bash ./train_scripts/train_i2v.sh
```

### Inference
You can use the provided ./test.sh script for inference. Provide a reference image or video (extract the first frame) to the asset/test.txt file and pass it to the --txt_file parameter.
```
CUDA_VISIBLE_DEVICES=0 python scripts/inference_i2v.py \
      --config=./configs/mobilei2v_config/MobileI2V_300M_img512.yaml \
      --save_path=humface_1126 \
      --model_path=./model/hybrid_371.pth \
      --txt_file=asset/test.txt \
      --flow_score=2.0 \
```
To achieve faster VAE decoder speeds, we replaced the LTX-Video decoder with the [Turbo-VAED](https://github.com/hustvl/Turbo-VAED) decoder.

### Metrics
Refer to the FVD evaluation script in [vidm](https://github.com/MKFMIKU/vidm/tree/main).
```
python scripts/evaluate_FVD.py -dir1 path/gts -dir2 path/videos -b 1 -r 32 -n 128 -ns 16 -i3d ./i3d_torchscript.pt
```

### Distillation training
The training data is consistent with that used for base model training. Please refer to our [distillation code](https://github.com/hustvl/MobileI2V/tree/distillation), which is available in the distillation branch of this repository.

## 📱 On-device Deployment

- **Core ML conversion:** [deployment/coreml](./deployment/coreml) provides the hybrid transformer and Turbo-VAE decoder exporters with numerical validation. On an Apple Silicon Mac, install [the export dependencies](./deployment/coreml/requirements.txt) in a separate Python 3.11/3.12 environment and run `python -m deployment.coreml.export --help` from the repository root.
- **iOS demo:** Open [MobileI2V.xcodeproj](./deployment/ios/MobileI2V.xcodeproj) in Xcode 16+ (iOS 18+). Add matching `Transformer`, `VAEEncoder`, and `VAEDecoder` Core ML packages to `deployment/ios/MobileI2V/Models/`, then select your own signing team to run on an iPhone.

Model weights are not bundled. The exporters require matching Diffusers-format transformer and Turbo-VAE checkpoints; the first-frame VAE encoder export is not included yet.

## 🎯 Mobile Demo
We designed the mobile UI and deployed the model, as shown in the video below:
<div align="center">
  <img src="./asset/videos/mobileUI.gif" width="25.5%">

</div>

## ❤️ Acknowledgements

Our MobileI2V codes are mainly built with [SANA](https://github.com/NVlabs/Sana) and [LTX-Video](https://github.com/Lightricks/LTX-Video). The data processing workflow is based on [OpenSora](https://github.com/hpcaitech/Open-Sora). Thanks for all these great works.




## 📝 Citation

If you find MobileI2V useful, please consider giving us a star 🌟 and citing it as follows:

```
@misc{MobileI2V,
      title={MobileI2V: Fast and High-Resolution Image-to-Video on Mobile Devices}, 
      author={Shuai Zhang and Bao Tang and Siyuan Yu and Yueting Zhu and Jingfeng Yao and Ya Zou and Shanglin Yuan and Li Yu and Wenyu Liu and Xinggang Wang},
      year={2025},
      eprint={2511.21475},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.21475}, 
}

```
