import os
import glob
import torch
import random
import argparse
import skimage.io
import numpy as np

from ssr.utils.infer_utils import format_s2naip_data
from ssr.utils.options import yaml_load
from ssr.utils.model_utils import build_network


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help="Path to the options file.")
    args = parser.parse_args()

    device = torch.device('cuda')

    # Load the configuration file.
    opt = yaml_load(args.opt)

    data_dir = opt['data_dir']  # root directory containing the low-res images you want to super-resolve
    n_lr_images = opt['n_lr_images']  # number of low-res images as input to the model; must be the same as when the model was trained
    save_path = opt['save_path']  # directory where model outputs will be saved

    # Define the generator model, based on the type and parameters specified in the config.
    model = build_network(opt)

    # Load the pretrained weights into the model
    if not 'pretrain_network_g' in opt['path']:
        print("WARNING: Model weights are not specified in configuration file.")
    else:
        weights = opt['path']['pretrain_network_g']  # path to the generator weights
        state_dict = torch.load(weights)
        model.load_state_dict(state_dict[opt['path']['param_key_g']], strict=opt['path']['strict_load_g'])
    model = model.to(device).eval()

    # The images in the data_dir for inference should be pngs or jpgs and the directory
    # structure should look like: {data_dir}/sentinel2/{subdir}/*.png or *.jpg.
    img_paths = []
    for ext in ['png', 'PNG', 'jpg', 'JPG', 'jpeg', 'JPEG']:
        img_paths.extend(glob.glob(os.path.join(data_dir, f"**/*.{ext}"), recursive=True))
    img_paths.sort()
    print("Running inference on ", len(img_paths), " images.")

    for png in img_paths:
        rel_path = os.path.relpath(png, data_dir)
        save_dir = os.path.join(save_path, os.path.dirname(rel_path))
        os.makedirs(save_dir, exist_ok=True)

        im = skimage.io.imread(png)

        # Feed the low-res images through the super-res model.
        input_tensor, s2_image = format_s2naip_data(im, n_lr_images, device)
        output = model(input_tensor)

        # Save the low-res input image in the same dir as the super-res image so
        # it is easy for the user to compare.
        base_name = os.path.basename(rel_path)
        lr_path = os.path.join(save_dir, f"lr_{base_name}")
        skimage.io.imsave(lr_path, s2_image)

        # Convert the model output back to a numpy array and adjust shape and range.
        output = torch.clamp(output, 0, 1)
        output = output.squeeze().cpu().detach().numpy()
        output = np.transpose(output, (1, 2, 0))  # transpose to [h, w, 3] to save as image
        output = (output * 255).astype(np.uint8)

        # Save the super-res output image with the same basename as the input.
        sr_path = os.path.join(save_dir, base_name)
        skimage.io.imsave(sr_path, output, check_contrast=False)


if __name__ == "__main__":
    main()

