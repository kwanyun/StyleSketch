# Stylized Face Sketch Extraction via Generative Prior with Limited Data (EUROGRAPHICS 2024)

### [EG2024] Official repository of StyleSketch [[SKSF-A](https://github.com/kwanyun/SKSF-A)] [[Project Page](https://kwanyun.github.io/stylesketch_project/)] [[Paper](https://arxiv.org/abs/2403.11263)]
![teaser2](https://github.com/kwanyun/StyleSketch/assets/68629563/e5368677-fbd4-4942-9385-ed7cc14de603)

### Getting Started
* install dependency
```bash
bash run.sh
```
* Put styleGAN related checkpoints folder in stylesketch/sketch folder
  ex) stylesketch/sketch/checkpoints/stylegan_pretrain

  https://drive.google.com/file/d/1X--a491Q6reEBV50XfyYqQ86yDxI44nd/view?usp=drive_link


* Put pretrained StyleSketch weights in model_dir
  ex) stylesketch/sketch/model_dir
  https://drive.google.com/file/d/17AgaRzSwXi3c5tmTZztrGGifyHGKrrQu/view?usp=drive_link


### How to inference Scripts
Move to sketch folder and run generate.py with the style to extract
```bash
cd sketch
python generate.py --train_data sketch_MJ
python generate.py --train_data pencil_sj
```
### How to make the w^+ code to extract sketches?
In our experiment, we used [e4e](https://github.com/omertov/encoder4editing) followed by optimization. This can be replaced by different inversion methods.


### SKSF-A Sketch Data
SKSF-A consists of seven distinct styles drawn by professional artists, each containing 134 identities and corresponding sketches.

### [SKSF-A](https://github.com/kwanyun/SKSF-A)

### Acknowledgments
our codes were borrowed from [DatasetGAN](https://github.com/nv-tlabs/datasetGAN_release)

### If you use this code or SKSF-A for your research, please cite our paper:
```bash
@article {yun2024stylized,
journal = {Computer Graphics Forum},
title = {{Stylized Face Sketch Extraction via Generative Prior with Limited Data}},
author = {Yun, Kwan and Seo, Kwanggyoon and Seo, Chang Wook and Yoon, Soyeon and Kim, Seongcheol and Ji, Soohyun and Ashtari, Amirsaman and Noh, Junyong},
year = {2024},
publisher = {The Eurographics Association and John Wiley & Sons Ltd.},
ISSN = {1467-8659},
DOI = {10.1111/cgf.15045}
}
```
