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

def create_video(img_array, video_dir, video_name, fps=30): 
    print(len(img_array))
    print(img_array[0].shape)
    height, width,c = img_array[0].shape

    size = (width,height)

    out = cv2.VideoWriter(f'{video_dir}/{video_name}.mp4',cv2.VideoWriter_fourcc('m', 'p','4', 'v'), int(fps), size)
    print(f"Video is saved as  {video_dir}/{video_name}.mp4")

    for i in range(len(img_array)):
        # print(i)
        out.write(img_array[i])
    
    out.release()
    print("done!")

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


if __name__ == '__main__':

    args = parse_args()

    # Create folder to save results
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    
    # If input folder is empty or not exist, exit
    if not os.path.exists(args.img_folder):
        print('Input folder does not exist!')
        exit()


    # Clear save folder
    # for file in os.listdir(args.save_folder):
    #     os.remove(os.path.join(args.save_folder, file))

    #If debug mode, wait for debugger to attach
    if args.debug:
        args = parse_args()
        debugpy.listen(5678)
        print("Press play!")
        debugpy.wait_for_client()


    # create depth model
    depth_model = RelDepthModel(backbone=args.backbone)
    depth_model.eval()

    # load checkpoint
    load_ckpt(args, depth_model, None, None)
    depth_model.cuda()


    # load images
    video_file_path = args.img_folder
    # image_dir = os.path.dirname(os.path.dirname(__file__)) + '/test_images/'
    # imgs_list = os.listdir(image_dir)
    # imgs_list.sort()

    print(f"Video input : {video_file_path}")
        # os.makedirs(os.path.join(args.outdir,os.path.basename(video_file_path)[:-4]), exist_ok=True)

    vidcap = cv2.VideoCapture(video_file_path)

    success, frame = vidcap.read()

    fps = int(vidcap.get(cv2.CAP_PROP_FPS))
    count = 0
    result_frame = []


    # If debug mode, only process 10 images
    # if args.debug:
    #     imgs_list = imgs_list[:10]


    # imgs_path = [os.path.join(args.img_folder, i) for i in imgs_list if i != 'outputs']


    while success:
        success, rgb = vidcap.read()
        if not success:
            break
        count += 1
        if count % 3 == 0:
            rgb_c = rgb[:, :, ::-1].copy()
            gt_depth = None

            # rgb_c= cv2.flip(rgb_c, 1)
            A_resize = cv2.resize(rgb_c, (448, 448))
            # print(A_resize.shape)
            # A_resize =flip_reshape_pad(rgb_c,False,(448, 448))
            # cv2.imwrite(os.path.join(args.save_folder,"origin_"+  v.split('/')[-1][:-4]+'.png'), A_resize)

            # print(A_resize.shape)
            rgb_half = cv2.resize(rgb, (rgb.shape[1]//2, rgb.shape[0]//2), interpolation=cv2.INTER_LINEAR)

            
            img_torch = scale_torch(A_resize)[None, :, :, :]
            
            start = time.time()
        
            pred_depth = depth_model.inference(img_torch).cpu().numpy().squeeze()
            end = time.time()
            
            pred_depth=pred_depth[6:pred_depth.shape[0]-6,6:pred_depth.shape[1]-6]
            pred_depth_ori = cv2.resize(pred_depth, (rgb.shape[1], rgb.shape[0]))
            # depth= (pred_depth_ori/pred_depth_ori.max() * 60000).astype(np.uint16)
            depth_color=255- ((pred_depth_ori - pred_depth_ori.min()) / (pred_depth_ori.max() - pred_depth_ori.min()) * 255.0)
            depth_color = cv2.applyColorMap(depth_color.astype(np.uint8), cv2.COLORMAP_INFERNO)
            # image = cv2.cvtColor(depth,cv2.COLOR_GRAY2BGR)
            # cvuint8 = cv2.convertScaleAbs(image)
            if len(result_frame) !=0:
                pre_depth_frame=result_frame[-1]
            else:
                pre_depth_frame=depth_color
            # alpha=0.1
            # beta=1-alpha
            # # print(type(depth_color),type(pre_depth_frame))
            # depth_color2=cv2.addWeighted(depth_color, alpha, pre_depth_frame, beta, 0)

            # result_frame.append(depth_color2)
            result_frame.append(depth_color)
            # result_frame.append(depth)

    create_video(result_frame, args.save_folder, f"{os.path.basename(video_file_path[:-4]) + '_AnyD_merge_pre2'}", fps=30)

        