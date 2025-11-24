<div align="center">

<p align="center" style="border-radius: 10px">
  <img src="asset/logo3.png" width="35%" alt="logo"/>
</p>

<h1>MobileI2V: Fast and High-Resolution Image-to-Video on Mobile Devices</h1>


[Shuai Zhang](https://github.com/Shuaizhang7)<sup>\*</sup>, Bao Tang<sup>\*</sup>, Siyuan Yu<sup>\*</sup>, Yueting Zhu, [Jingfeng Yao](https://github.com/JingfengYao),<br>Ya Zou, Shanglin Yuan, Li Yu, [Wenyu Liu](http://eic.hust.edu.cn/professor/liuwenyu), [Xinggang Wang](https://xwcv.github.io/index.htm)<sup>📧</sup>


Huazhong University of Science and Technology (HUST) 

(\* equal contribution, 📧 corresponding author)

[![arxiv paper](https://img.shields.io/badge/arXiv-Paper-red)](https://arxiv.org/abs/2508.09136)
[![checkpoints](https://img.shields.io/badge/HuggingFace-🤗-green)](https://huggingface.co/hustvl/Turbo-VAED)

</div>

## 📰 News
- **[2025.11.25]** We have released our paper on [arXiv](https://arxiv.org/abs/2508.09136).

## 📄 Introduction
<div align="center">
<img src="./asset/t1_4.pdf">
</div>
Recently, video generation has witnessed rapid advancements, drawing increasing attention to image-to-video (I2V) synthesis on mobile devices. However, the substantial computational complexity and slow generation speed of diffusion models pose significant challenges for real-time, high-resolution video generation on resource-constrained mobile devices. In this work, we propose MobileI2V, a 270M lightweight diffusion model for real-time image-to-video generation on mobile devices. The core lies in: (1) We analyzed the performance of linear attention modules and softmax attention modules on mobile devices, and proposed a linear hybrid architecture denoiser that balances generation efficiency and quality. (2) We design a time-step distillation strategy that compresses the I2V sampling steps from more than 20 to only two without significant quality loss, resulting in a 10-fold increase in generation speed. (3) We apply mobile-specific attention optimizations that yield 2$\times$ speed-up for attention operations during on-device inference. MobileI2V enables, for the first time, fast 720p image-to-video generation on mobile devices, with quality comparable to existing models. Under one-step conditions, the generation speed of each frame of 720p video is less than 100 ms. Our code will be publicly released.



## 🎯 How to Use

### Installation

```

```



## ❤️ Acknowledgements

Our MobileI2V codes are mainly built with [SANA](https://github.com/NVlabs/Sana) and [LTX-VIdeo](https://github.com/Lightricks/LTX-Video). Thanks for all these great works.

## 📝 Citation

If you find MobileI2V useful, please consider giving us a star 🌟 and citing it as follows:

```


```
