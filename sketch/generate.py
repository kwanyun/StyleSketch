
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
import json
import torch
import torch.nn as nn
import numpy as np

from stylesketch_utils.stylesketch import SketchGenerator,Discriminator
from stylesketch_utils.prepare_stylegan import prepare_stylegan
from PIL import Image
from utils.utils import latent_to_image, oht_to_scalar_regression
from tqdm import tqdm
import argparse
import glob


device = 'cuda' if torch.cuda.is_available() else 'cpu'
device_ids = [0,1]


def parallelize(model):
    """
    Distribute a model across multiple GPUs.
    """
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model, device_ids=device_ids)
    return model


def generate_data(args, name_train_style="Disney_sketch_MJ"):
    checkpoint_path = args['stylesketch_path']
    result_path = os.path.join(checkpoint_path, 'samples' )
    
    if os.path.exists(result_path):
        pass
    else:
        os.system('mkdir -p %s' % (result_path))
        print('Experiment folder created at: %s' % (result_path))
        
    g_all, avg_latent = prepare_stylegan(args['stylegan_checkpoint'],args['saved_latent'])
    
    
    generator = SketchGenerator()
    generator = generator.cuda()
    generator = parallelize(generator)
    
    checkpoint = torch.load(os.path.join(checkpoint_path, f'model_{name_train_style}.pth'))
    generator.load_state_dict(checkpoint['model_state_dict'],strict =True)
    generator.eval()

    with torch.no_grad():
        #get sd2 latents to sketch
        latents_to_sketch = glob.glob(os.path.join(args['image_latent_path'],'*'))
        latents_to_sketch = [latent for latent in latents_to_sketch if latent.endswith('pt')]
        print( "num_sketches: ", len(latents_to_sketch))

        for latent_dir in tqdm(latents_to_sketch):
            single_latent = torch.load(latent_dir)
            latent_input = single_latent.float().to(device)
            latent_name = latent_dir.split('/')[-1].split('.')[0]
            #extract features from sd model
            img, affine_layers = latent_to_image(g_all, latent_input, 3,dim=args['dim'][1],use_style_latents=args['annotation_data_from_w'])
            affine_layers.append(img.transpose(0,3,1,2))
            affine_layers = [torch.from_numpy(x).type(torch.FloatTensor).to(device) for x in affine_layers]
            
            #generate sketch from features
            sketch_image = generator(affine_layers)
            
            #save sketch
            sketch_image = oht_to_scalar_regression(sketch_image.squeeze())
            sketch_image = sketch_image.cpu().detach().numpy()
            image_label_name = os.path.join(result_path, f'Sketch_{latent_name}_{name_train_style}.png')
            sketch = Image.fromarray(sketch_image.astype('uint8'))
            sketch.save(image_label_name)
            
            #save images corresponding to sketch
            img = Image.fromarray(np.asarray(img.squeeze()))
            image_name_ori = os.path.join(result_path,f'Image_{latent_name}.png')
            img.save(image_name_ori)
            
            
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default = "experiments/inference.json")
    parser.add_argument('--train_data', type=str)
    
    args = parser.parse_args()
    opts = json.load(open(args.exp, 'r'))    
    
    
    
    generate_data(opts, args.train_data)