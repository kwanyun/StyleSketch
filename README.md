# Stylized Face Sketch Extraction via Generative Prior with Limited Data (EUROGRAPHICS 2024)

### Official repository of StyleSketch
[DATA SET](https://github.com/kwanyun/SKSF-A) &nbsp;&nbsp; [Project Page]()
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
python generate.py --train_data model_sketch_MJ
python generate.py --train_data model_pencil_sj
```
### How to get the pt file
In our experiment, we used [e4e](https://github.com/omertov/encoder4editing) followed by optimization. This can be replaced by different inversion methods.


### SKSF-A Sketch Data
SKSF-A consists of seven distinct styles drawn by professional artists. SKSF-A contains 134 identities and corresponding sketches, making a total of 938 face-sketch pairs.

### [SKSF-A](https://github.com/kwanyun/SKSF-A)

### Acknowledgments
our codes were borrowed from [DatasetGAN](https://github.com/nv-tlabs/datasetGAN_release)

### If you use this code or SKSF-A for your research, please cite our paper:
```bash
@article{yun2024stylized,
  title={Stylized Face Sketch Extraction via Generative Prior with Limited Data},
  author={anon},
  journal={anon},
  year={2024}
}
```
