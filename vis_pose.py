import os
import mmcv
import argparse
import glob
import sys
import numpy as np
import cv2
sys.path.append('..')
from ref import lm_full
from tqdm import tqdm
from render_xyz.model import load_models
from render_xyz.render_xyz import Renderer
from core.gdrn_modeling.models import GDRN, GDRNT
import torch 
#model = torch.load("/home/khiemphi/GDR-Net/output/gdrn/lm/a6_cPnP_lm13/model_0133249.pth")
#print(model)
#breakpoint()
from core.utils.my_checkpoint import MyCheckpointer

parser = argparse.ArgumentParser("Visualize qualitative results on LM by rendering the CAD models")
parser.add_argument("-c", type=str, help="the config file")
parser.add_argument("-vis_h", type=int, default=256, help="the height of final output image")
parser.add_argument("-vis_w", type=int, default=256, help="the width of final output image")
parser.add_argument("-num_per_obj", type=int, default=20, help="the number of images per object")
args = parser.parse_args()
os.chdir('..')

# parse the config file and find the output root
cfg = mmcv.Config.fromfile(args.c)
output_root = "/home/khiemphi/GDR-Net/vis"
# read in the prediction
pred_path = '/home/khiemphi/GDR-Net/output/gdrn/lm/a6_cPnP_lm13/inference_epoch_9_iter_127999/lm_13_test/a6-cPnP-lm13-T_lm_13_test_preds.pkl'
pred_all = mmcv.load(pred_path)

pred_path_baseline = "/home/khiemphi/GDR-Net/output/gdrn/lm/a6_cPnP_lm13/inference_gdrn_lm/lm_13_test/a6-cPnP-lm13-test_lm_13_test_preds.pkl"
pred_baseline = mmcv.load(pred_path_baseline)

model, optimizer = eval(cfg.MODEL.CDPN.NAME).build_model_optimizer(cfg)


MyCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load('home/khiemphi/GDR-Net/output/gdrn/lm/a6_cPnP_lm13/model_final.pth', resume=False)


ren = Renderer(size=(lm_full.width, lm_full.height), cam=lm_full.camera_matrix)
ren_models = load_models(
    model_paths=lm_full.model_paths,
    vertex_scale=0.001,
    center=False,
    obj_ids=lm_full.id2obj
)




# read in the object models used in LM dataset

# objects used in LM experiments
LM_13_OBJECTS = [
    "ape",
    "benchvise",
    "camera",
    "can",
    "cat",
    "driller",
    "duck",
    "eggbox",
    "glue",
    "holepuncher",
    "iron",
    "lamp",
    "phone",
]

def depth2contour(depth):
    mask = (depth > 0)
    mask_threshold = mask.astype(np.uint8) * 255
    contours, hierarchy = cv2.findContours(mask_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    return contours, hierarchy


# xyxy
def get_bbox_from_mask(mask):
    vu = np.where(mask)
    if (len(vu[0]) > 0):
        return np.array([np.min(vu[1]), np.min(vu[0]), np.max(vu[1]), np.max(vu[0])], np.int64)
    else:
        return np.zeros((4), np.int)

# cropped the image
# need to contain the object
def crop_image(src_img, depth1, depth2, h, w, H=480, W=640):
    mask = np.logical_or(depth1>0, depth2>0)
    x1, y1, x2, y2 = get_bbox_from_mask(mask)
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    # first, try to crop by putting the bounding box at the center
    nx1 = max(cx - w // 2, 0)
    ny1 = max(cy - h // 2, 0)
    nx2 = nx1 + w
    ny2 = ny1 + h
    if nx2 > W or ny2 > H:
        nx2 = min(W, nx2)
        ny2 = min(H, ny2)
        nx1 = nx2 - w
        ny1 = ny2 - h
    cropped = src_img[ny1:ny2, nx1:nx2, :]
    cropped = np.array(cropped, dtype=np.int64)
    return cropped




pbar = tqdm(LM_13_OBJECTS)
for obj_name in pbar:
    obj_id = lm_full.obj2id[obj_name]
    im_id_file = "/home/khiemphi/GDR-Net/datasets/BOP_DATASETS/lm/image_set/{}_test.txt".format(obj_name)
    with open(im_id_file, "r") as f:
        indices = [line.strip("\r\n") for line in f.readlines()]  # string ids
        # select an id
        for ii in range(args.num_per_obj):
            im_id = indices[ii]  # already 6-digit id
            folder = '/home/khiemphi/GDR-Net/'
            im_full_path = 'datasets/BOP_DATASETS/lm/test/{:06d}/rgb/{}.png'.format(obj_id, im_id)

            pbar.set_postfix({"object": obj_name, "imd_id": im_id})
            #print("object: {} im_id: {}".format(obj_name, im_id))
            
            im_ori = mmcv.imread(os.path.join(folder, im_full_path), "color")
            # read in the corresponding GT and compute the contour
            gt_path = '/home/khiemphi/GDR-Net/datasets/BOP_DATASETS/lm/test/{:06d}/scene_gt.json'.format(obj_id)
            gt_all = mmcv.load(gt_path)
            im_id_short = "0" if im_id == "000000" else im_id.lstrip("0")
            gt = gt_all[im_id_short][0]
            gt_rot = np.array(gt["cam_R_m2c"], dtype=np.float32).reshape(3, 3)
            gt_trans = np.array(gt["cam_t_m2c"], dtype=np.float32) / 1000.0
            gt_pose = np.eye(4)
            gt_pose[:3, :3] = gt_rot
            gt_pose[:3, 3] = gt_trans

            ren.clear()
          
            ren.draw_model(ren_models[obj_name], gt_pose)
           
            _, gt_depth = ren.finish()
            gt_cont, _ = depth2contour(gt_depth)
            # find the corresponding prediction
            pred = pred_all[obj_name][im_full_path]
            pred_rot = pred['R']
            pred_t = pred['t']
            pred_pose = np.eye(4)
            pred_pose[:3, :3] = pred_rot
            pred_pose[:3, 3] = pred_t
            
            baseline = pred_baseline[obj_name][im_full_path]
            baseline_rot = baseline['R']
            baseline_t = baseline['t']
            baseline_pose = np.eye(4)
            baseline_pose[:3, :3] = baseline_rot
            baseline_pose[:3, 3] = baseline_t

            
            
            
            # render the objects and draw the contour
            ren.clear()
            ren.draw_background(im_ori)
            ren.draw_model(ren_models[obj_name], pred_pose)
            ren_im, ren_depth = ren.finish()
            
            ren.clear()
            ren.draw_background(im_ori)
            ren.draw_model(ren_models[obj_name], baseline_pose)
            ren_im_baseline, ren_depth_baseline = ren.finish()



            ren_im = ren_im * 255
            pred_cont, _ = depth2contour(ren_depth)
            baseline_cont, _ = depth2contour(ren_depth_baseline)

            ren_im = cv2.drawContours(ren_im, gt_cont, -1, (255, 0, 0), 2)  # render gt first # v
            ren_im = cv2.drawContours(ren_im, pred_cont, -1, (0, 255, 0), 2) # g
            ren_im = cv2.drawContours(ren_im, baseline_cont, -1, (0, 0, 255), 2) #r         
            
            
            # save cropped image
            save_path = os.path.join(output_root, "lm_qual", "render_{}_{}.png".format(obj_name, im_id))
            mmcv.mkdir_or_exist(os.path.dirname(save_path))
            ren_im_crop = crop_image(ren_im, ren_depth, gt_depth, args.vis_h, args.vis_w)
            
            mmcv.imwrite(ren_im_crop, save_path)
            print("rendered result saved to: {}".format(save_path))

            
           