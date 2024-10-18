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
import torch.nn as nn
from lib.multi_depth_model_woauxi import RelDepthModel
# from lib.net_tools import load_ckpt
from lib.net_tools import get_func
from lib import network_auxi as network
class DepthModel(nn.Module):
    def __init__(self, encoder):
        super(DepthModel, self).__init__()
        backbone = network.__name__.split('.')[-1] + '.' + encoder
        self.encoder_modules = get_func(backbone)()
        self.decoder_modules = network.Decoder()

    def forward(self, x):
        lateral_out = self.encoder_modules(x)
        out_logit = self.decoder_modules(lateral_out)
        return out_logit

def load_ckpt(args, depth_model, shift_model, focal_model):
    """
    Load checkpoint.
    """
    if os.path.isfile(args.load_ckpt):
        print("loading checkpoint %s" % args.load_ckpt)
        checkpoint = torch.load(args.load_ckpt,map_location='cpu')
        # print(checkpoint['model_state_dict'].keys())
        if shift_model is not None:
            shift_model.load_state_dict(strip_prefix_if_present(checkpoint['shift_model'], 'module.'),
                                    strict=True)
        if focal_model is not None:
            focal_model.load_state_dict(strip_prefix_if_present(checkpoint['focal_model'], 'module.'),
                                    strict=True)
        try:   
            print("error")                         
            depth_model.load_state_dict(strip_prefix_if_present(checkpoint['model_state_dict'], "module."),
                                    strict=False)
        except Exception:
            try:
                depth_model.load_state_dict(strip_prefix_if_present(checkpoint['depth_model'], "module"),
                                        strict=True) 
            except Exception:
                try:
                    depth_model.load_state_dict(strip_prefix_if_present(checkpoint['model_state_dict'], "module.depth_model"),
                                        strict=False)
                except Exception:
                    depth_model.load_state_dict(strip_prefix_if_present(checkpoint['depth_model'], "module.depth_model"),
                                        strict=True)
        del checkpoint
        torch.cuda.empty_cache()


def strip_prefix_if_present(state_dict, prefix):
    from collections import OrderedDict

    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict

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
    if len(os.listdir(args.img_folder)) == 0:
        print('Input folder is empty!')
        exit()

    # Clear save folder
    for file in os.listdir(args.save_folder):
        os.remove(os.path.join(args.save_folder, file))

    #If debug mode, wait for debugger to attach
    if args.debug:
        args = parse_args()
        debugpy.listen(5678)
        print("Press play!")
        debugpy.wait_for_client()


    # create depth model
    # depth_model = RelDepthModel(backbone=args.backbone)
    depth_model = DepthModel('resnet34_stride32')
    depth_model.eval()
    print(depth_model)
    # load checkpoint
    load_ckpt(args, depth_model, None, None)
    # depth_model.cuda()


    # load images
    imgs_list = os.listdir(args.img_folder)
    # image_dir = os.path.dirname(os.path.dirname(__file__)) + '/test_images/'
    # imgs_list = os.listdir(image_dir)
    imgs_list.sort()



    # If debug mode, only process 10 images
    if args.debug:
        imgs_list = imgs_list[:10]


    imgs_path = [os.path.join(args.img_folder, i) for i in imgs_list if i != 'outputs']

    base_name=args.load_ckpt.split("/")[-2]+"_"+args.load_ckpt.split("/")[-1][:-4]
    for i, v in enumerate(imgs_path):
        print('processing (%04d)-th image... %s' % (i, v))
        rgb = cv2.imread(v)
        rgb_c = rgb[:, :, ::-1].copy()
        gt_depth = None
        A_resize = cv2.resize(rgb_c, (448, 448))
        rgb_half = cv2.resize(rgb, (rgb.shape[1]//2, rgb.shape[0]//2), interpolation=cv2.INTER_LINEAR)

        img_torch = scale_torch(A_resize)[None, :, :, :]
        print(img_torch.shape,A_resize.shape)
        start = time.time()
    
        # pred_depth = depth_model.inference(img_torch).cpu().numpy().squeeze()
        pred_depth = depth_model(img_torch).detach().numpy().squeeze()
        pred_depth = pred_depth - pred_depth.min() + 0.01
        
        end = time.time()
        
        print("FPS: ", 1/(end-start))
        pred_depth_ori = cv2.resize(pred_depth, (rgb.shape[1], rgb.shape[0]))
        print(pred_depth_ori.max() )
        # if GT depth is available, uncomment the following part to recover the metric depth
        #pred_depth_metric = recover_metric_depth(pred_depth_ori, gt_depth)

        img_name = v.split('/')[-1]
        # cv2.imwrite(os.path.join(image_dir_out, img_name), rgb)
        # save depth
        # plt.imsave(os.path.join(image_dir_out, img_name[:-4]+'-depth.png'), pred_depth_ori, cmap='rainbow')
        cv2.imwrite(os.path.join(args.save_folder, img_name[:-4]+'.png'), (pred_depth_ori/pred_depth_ori.max() * 60000).astype(np.uint16))
        torch.onnx.export(depth_model,img_torch, args.save_folder+"/"+base_name+".onnx",opset_version=11,keep_initializers_as_inputs=True)
        exit()

    if args.create_video:
        create_video(args.save_folder, args.save_folder, args.video_name, args.fps)
        