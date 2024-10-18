import cv2
import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import time 
import debugpy 
import tqdm 
import glob 

from lib.multi_depth_model_woauxi import RelDepthModel
from lib.net_tools import load_ckpt


def add_create_video_args(parser):
    parser.add_argument('--create_video', action='store_true', help='create video', default=False)
    parser.add_argument('--video_name', default='video.mp4', help='video name')
    parser.add_argument('--fps', default=30, help='video fps')
    return parser



def parse_args():
    parser = argparse.ArgumentParser(
        description='Configs for LeReS')
    parser.add_argument('--load_ckpt', default='./res50.pth', help='Checkpoint path to load')
    parser.add_argument('--backbone', default='resnext101', help='Checkpoint path to load')
    parser.add_argument('--img_folder', default='./data/depth_test', help='Folder path to load')
    parser.add_argument('--save_folder', default='./data/output_folder', help='Folder path to save')
    parser.add_argument('--debug', action='store_true', help='debug mode', default=False)

    parser = add_create_video_args(parser)


    args = parser.parse_args()
    return args

def create_video(mypath, outputpath, name, fps=30): 
    size = None 
    img_array = []

    for filename in sorted(glob.glob(f'{mypath}/*')):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)

    out = cv2.VideoWriter(f'{outputpath}/{name}.mp4',cv2.VideoWriter_fourcc(*'MP4V'), int(fps), size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

def flip_reshape_pad(img, flip, resize_size, pad_value=0, resize_method='bilinear'):
        """
        Flip, pad, reshape, and crop the image.
        :param img: input image, [C, H, W]
        :param flip: flip flag
        :param crop_size: crop size for the image, [x, y, width, height]
        :param pad: pad the image, [up, down, left, right]
        :param pad_value: padding value
        :return:
        """
        pad=[6,6,6,6]
        # Flip
        if flip:
            img = np.flip(img, axis=1)
        

        # Pad the raw image
        if len(img.shape) == 3:
            img_pad = np.pad(img, ((pad[0], pad[1]), (pad[2], pad[3]), (0, 0)), 'constant',
                             constant_values=(pad_value, pad_value))
        else:
            img_pad = np.pad(img, ((pad[0], pad[1]), (pad[2], pad[3])), 'constant',
                             constant_values=(pad_value, pad_value))
        # Resize the raw image
        if resize_method == 'nearest':
            img_resize = cv2.resize(img_pad, (resize_size[1], resize_size[0]), interpolation=cv2.INTER_NEAREST)
        else:
            img_resize = cv2.resize(img_pad, (resize_size[1], resize_size[0]), interpolation=cv2.INTER_LINEAR)

        # # Resize the raw image
        # if resize_method == 'bilinear':
        #     img_resize = cv2.resize(img, (resize_size[1], resize_size[0]), interpolation=cv2.INTER_LINEAR)
        # elif resize_method == 'nearest':
        #     img_resize = cv2.resize(img, (resize_size[1], resize_size[0]), interpolation=cv2.INTER_NEAREST)
        # else:
        #     raise ValueError

        # # Crop the resized image
        # img_crop = img_resize[crop_size[1]:crop_size[1] + crop_size[3], crop_size[0]:crop_size[0] + crop_size[2]]

        # # Pad the raw image
        # if len(img.shape) == 3:
        #     img_pad = np.pad(img_crop, ((pad[0], pad[1]), (pad[2], pad[3]), (0, 0)), 'constant',
        #                      constant_values=(pad_value, pad_value))
        # else:
        #     img_pad = np.pad(img_crop, ((pad[0], pad[1]), (pad[2], pad[3])), 'constant',
        #                      constant_values=(pad_value, pad_value))

        return img_resize

def scale_torch(img):
    """
    Scale the image and output it in torch.tensor.
    :param img: input rgb is in shape [H, W, C], input depth/disp is in shape [H, W]
    :param scale: the scale factor. float
    :return: img. [C, H, W]
    """
    if len(img.shape) == 2:
        img = img[np.newaxis, :, :]
    if img.shape[2] == 3:
        transform = transforms.Compose([transforms.ToTensor(),
		                                transforms.Normalize((0.485, 0.456, 0.406) , (0.229, 0.224, 0.225) )])
        img = transform(img)
    else:
        img = img.astype(np.float32)
        img = torch.from_numpy(img)
    return img















def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == '__main__':

    args = parse_args()


    # create depth model
    depth_model = RelDepthModel(backbone=args.backbone)
    depth_model.eval()

    # load checkpoint
    load_ckpt(args, depth_model, None, None)
    depth_model.cuda()


    #dummy_input = torch.randn(3,640,640, dtype=torch.float)
    dummy_input = np.random.randint(255, size=(640,640, 3), dtype=np.uint8)
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 100
    rgb_c = dummy_input[:, :, ::-1].copy()
    gt_depth = None
    A_resize = cv2.resize(rgb_c, (640, 640))
    rgb_half = cv2.resize(dummy_input, (dummy_input.shape[1]//2, dummy_input.shape[0]//2), interpolation=cv2.INTER_LINEAR)
    timings = np.zeros((repetitions,1))

    img_torch = scale_torch(A_resize)[None, :, :, :]

    # WARM-UP GPU
    for _ in range(10):
        _ = depth_model.inference(img_torch).cpu().numpy().squeeze()


    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()

            #_ = model(dummy_input)
            pred_depth = depth_model.inference(img_torch)

            ender.record()
            pred_depth = pred_depth.cpu().numpy().squeeze()

            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    print(mean_syn, f"NUM OF Param : {count_parameters(depth_model)}")
        