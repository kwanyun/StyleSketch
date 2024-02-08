# Stylized Face Sketch Extraction via Generative Prior with Limited Data (EUROGRAPHICS 2024)

## Official repository of StyleSketch

## Getting Started
* install dependency
```bash
bash setup.sh
```
* Put styleGAN related checkpoints folder in stylesketch/sketch folder
  * ex) stylesketch/sketch/checkpoints/stylegan_pretrain
```bash
https://drive.google.com/file/d/1pvyW_I-J0dMPqPnnYBcBeeevAI1O-wam/view?usp=drive_link
https://drive.google.com/file/d/1X--a491Q6reEBV50XfyYqQ86yDxI44nd/view?usp=drive_link
```


* Put pretrained StyleSketch weights in model_dir
  * ex) stylesketch/sketch/model_dir
  * `https://drive.google.com/file/d/17AgaRzSwXi3c5tmTZztrGGifyHGKrrQu/view?usp=drive_link`


## How to inference Scripts
```bash
cd sketch
python generate.py --train_data model_sketch_MJ
python generate.py --train_data model_pencil_sj
```


# SKSF-A Sketch Data
SKSF-A consists of seven distinct styles drawn by professional artists. SKSF-A contains 134 identities and corresponding sketches, making a total of 938 face-sketch pairs.
```bash
https://github.com/kwanyun/SKSF-A
```
