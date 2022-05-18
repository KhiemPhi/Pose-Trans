import logging
import os
import os.path as osp
import time

import numpy as np
import torch
from detectron2.checkpoint import PeriodicCheckpointer
from detectron2.data import MetadataCatalog
from detectron2.data.common import AspectRatioGroupedDataset
from detectron2.utils.events import EventStorage
from loguru import logger as loguru_logger
from pytorch_lightning import seed_everything

import core.utils.my_comm as comm
import ref
from core.gdrn_modeling.data_loader import (build_gdrn_test_loader,
                                            build_gdrn_train_loader)
from core.gdrn_modeling.dataset_factory import register_datasets_in_cfg
from core.gdrn_modeling.engine import GDRN_Lite
from core.gdrn_modeling.engine_utils import (batch_data, get_out_coor,
                                             get_out_mask)
from core.gdrn_modeling.main_gdrn import setup
from core.gdrn_modeling.models import GDRN, GDRNT
from core.utils import solver_utils
from core.utils.data_utils import denormalize_image
from core.utils.default_args_setup import (my_default_argument_parser,
                                           my_default_setup)
from core.utils.my_checkpoint import MyCheckpointer
from core.utils.my_setup import setup_for_distributed
from core.utils.my_writer import (MyCommonMetricPrinter, MyJSONWriter,
                                  MyPeriodicWriter, MyTensorboardXWriter)
from core.utils.utils import get_emb_show
from lib.utils.time_utils import get_time_str
from lib.utils.utils import dprint, get_time_str, iprint

logger = logging.getLogger("detectron2")
import datetime
from collections import OrderedDict

from detectron2.evaluation import (DatasetEvaluator, DatasetEvaluators,
                                   inference_context)
from detectron2.utils.logger import log_every_n_seconds
from torch.cuda.amp import autocast

from core.gdrn_modeling.gdrn_evaluator import (GDRN_Evaluator,
                                               gdrn_inference_on_dataset,
                                               save_result_of_dataset)
from core.utils.my_comm import (all_gather, get_world_size, is_main_process,
                                synchronize)
import cv2

def gdrn_inference_feat_on_dataset(cfg, model, data_loader, evaluator, amp_test=False):
    """Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.forward` accurately. The model
    will be used in eval mode.

    Args:
        model (nn.Module): a module which accepts an object from
            `data_loader` and returns some outputs. It will be temporarily set to `eval` mode.

            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator (DatasetEvaluator): the evaluator to run. Use `None` if you only want
            to benchmark, but don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} images".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_compute_time = 0
    total_process_time = 0
    eval_objs = []

    with inference_context(model), torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
                total_process_time = 0

            start_compute_time = time.perf_counter()
            #############################
            # process input
            if not isinstance(inputs, list):  # bs=1
                inputs = [inputs]
            batch = batch_data(cfg, inputs, phase="test")
            if evaluator.train_objs is not None:
                roi_labels = batch["roi_cls"].cpu().numpy().tolist()
                obj_names = [evaluator.obj_names[_l] for _l in roi_labels]
                if all(_obj not in evaluator.train_objs for _obj in obj_names):
                    continue
            
            if len(eval_objs) == 0 or obj_names[0] not in eval_objs: 
                
                eval_objs.append(obj_names[0])
            else: 
                
                continue

            with autocast(enabled=amp_test):
            
                if cfg.MODEL.WEIGHTS == '/home/khiemphi/GDR-Net/gdrn_lm_resnet.pth':
                    features = model(
                        batch["roi_img"],
                    )
                else: 
                    features = model(
                        batch["roi_img"],
                    )["last_hidden_state"].squeeze(0)

                features-= features.mean()
                features/= features.std ()
                features*=  64
                features+= 128
               
                if cfg.MODEL.WEIGHTS == '/home/khiemphi/GDR-Net/gdrn_lm_resnet.pth':
                    features_np = features.cpu().numpy()
                    features_np = np.resize( features_np,  (batch["roi_img"].shape[2:]))
                else: 
                    features_np = features.cpu().numpy().reshape((batch["roi_img"].shape[2:])) 
                
                features_np = np.clip(features_np, 0, 255).astype('uint8')
               
                blur = cv2.GaussianBlur(features_np,(5,5), 2).astype(np.uint8)
                heatmap_img = cv2.applyColorMap(blur, cv2.COLORMAP_JET)
                

                img = batch["roi_img"].squeeze(0).cpu().numpy().transpose(1, 2, 0)
                img *= 255  
                img = img.astype(np.uint8)
                super_imposed_img = cv2.addWeighted(heatmap_img, 0.5, img, 0.5, 0)

                if cfg.MODEL.WEIGHTS == '/home/khiemphi/GDR-Net/gdrn_lm_resnet.pth':
                    filename = "features_resnet_" + obj_names[0] + "_" + os.path.basename(inputs[0]["file_name"][0])
                    folder_path = "features_vis/resnet"
                else:
                    filename = "features_swin_" + obj_names[0] + "_" + os.path.basename(inputs[0]["file_name"][0])
                    folder_path = "features_vis/swin"
                full_path = os.path.join(folder_path, filename)

                cv2.imwrite(full_path, super_imposed_img)
                

                
           
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            cur_compute_time = time.perf_counter() - start_compute_time
            total_compute_time += cur_compute_time
            # NOTE: added
            # TODO: add detection time here
            outputs = [{} for _ in range(len(inputs))]
            for _i in range(len(outputs)):
                outputs[_i]["time"] = cur_compute_time

            start_process_time = time.perf_counter()
           
       
            cur_process_time = time.perf_counter() - start_process_time
            total_process_time += cur_process_time

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO, f"Inference done {idx+1}/{total}. {seconds_per_img:.4f} s / img. ETA={str(eta)}", n=5
                )

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        f"Total inference time: {total_time_str} "
        f"({total_time / (total - num_warmup):.6f} s / img per device, on {num_devices} devices)"
    )
    # pure forward time
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )
    # post_process time
    total_process_time_str = str(datetime.timedelta(seconds=int(total_process_time)))
    logger.info(
        "Total inference post process time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_process_time_str, total_process_time / (total - num_warmup), num_devices
        )
    )





class Lite(GDRN_Lite):
    def set_my_env(self, args, cfg):
        my_default_setup(cfg, args)  # will set os.environ["PYTHONHASHSEED"]
        seed_everything(int(os.environ["PYTHONHASHSEED"]), workers=True)
        setup_for_distributed(is_master=self.is_global_zero)
    
    def do_test_feat_map(self, cfg, model, epoch=None, iteration=None):
        results = OrderedDict()
        model_name = osp.basename(cfg.MODEL.WEIGHTS).split(".")[0]
        for dataset_name in cfg.DATASETS.TEST:
            if epoch is not None and iteration is not None:
                evaluator = self.get_evaluator(
                    cfg,
                    dataset_name,
                    osp.join(cfg.OUTPUT_DIR, f"inference_epoch_{epoch}_iter_{iteration}", dataset_name),
                )
            else:
                evaluator = self.get_evaluator(
                    cfg, dataset_name, osp.join(cfg.OUTPUT_DIR, f"inference_{model_name}", dataset_name)
                )
            data_loader = build_gdrn_test_loader(cfg, dataset_name, train_objs=evaluator.train_objs)
            data_loader = self.setup_dataloaders(data_loader, replace_sampler=False, move_to_device=False)
           
            backbone = model.module.backbone

            # Now we do forward with the backbone to get feature maps
            gdrn_inference_feat_on_dataset(cfg, backbone, data_loader, evaluator, amp_test=cfg.TEST.AMP_TEST)


            results_i = gdrn_inference_on_dataset(cfg, model, data_loader, evaluator, amp_test=cfg.TEST.AMP_TEST)
            
            
            results[dataset_name] = results_i

        if len(results) == 1:
            results = list(results.values())[0]
        return results

    def run(self, args, cfg):
        self.set_my_env(args, cfg)

        logger.info(f"Used GDRN module name: {cfg.MODEL.CDPN.NAME}")
        model, optimizer = eval(cfg.MODEL.CDPN.NAME).build_model_optimizer(cfg)

        #Let's load weight with the pytorch-way

        
        
        logger.info("Model:\n{}".format(model))

        # don't forget to call `setup` to prepare for model / optimizer for distributed training.
        # the model is moved automatically to the right device.
        model, optimizer = self.setup(model, optimizer) # look into setup to swap out features
        

        if True:
            # sum(p.numel() for p in model.parameters() if p.requires_grad)
            params = sum(p.numel() for p in model.parameters()) / 1e6
            logger.info("{}M params".format(params))

        if args.eval_only:
            MyCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
            self.do_test_feat_map(cfg, model)
            return self.do_test(cfg, model)

    
        self.do_train(cfg, args, model, optimizer, resume=args.resume)
        return self.do_test(cfg, model)
    
    def do_train(self, cfg, args, model, optimizer, resume=False):
        
        model.train()

        # some basic settings =========================
        dataset_meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        data_ref = ref.__dict__[dataset_meta.ref_key]
        obj_names = dataset_meta.objs

        # load data ===================================
        train_dset_names = cfg.DATASETS.TRAIN
        data_loader = build_gdrn_train_loader(cfg, train_dset_names)
        data_loader_iter = iter(data_loader)

        # load 2nd train dataloader if needed
        train_2_dset_names = cfg.DATASETS.get("TRAIN2", ())
        train_2_ratio = cfg.DATASETS.get("TRAIN2_RATIO", 0.0)
        if train_2_ratio > 0.0 and len(train_2_dset_names) > 0:
            data_loader_2 = build_gdrn_train_loader(cfg, train_2_dset_names)
            data_loader_2_iter = iter(data_loader_2)
        else:
            data_loader_2 = None
            data_loader_2_iter = None

        images_per_batch = cfg.SOLVER.IMS_PER_BATCH
        if isinstance(data_loader, AspectRatioGroupedDataset):
            dataset_len = len(data_loader.dataset.dataset)
            if data_loader_2 is not None:
                dataset_len += len(data_loader_2.dataset.dataset)
            iters_per_epoch = dataset_len // images_per_batch
        else:
            dataset_len = len(data_loader.dataset)
            if data_loader_2 is not None:
                dataset_len += len(data_loader_2.dataset)
            iters_per_epoch = dataset_len // images_per_batch
        max_iter = cfg.SOLVER.TOTAL_EPOCHS * iters_per_epoch
        dprint("images_per_batch: ", images_per_batch)
        dprint("dataset length: ", dataset_len)
        dprint("iters per epoch: ", iters_per_epoch)
        dprint("total iters: ", max_iter)

        data_loader = self.setup_dataloaders(data_loader, replace_sampler=False, move_to_device=False)
        if data_loader_2 is not None:
            data_loader_2 = self.setup_dataloaders(data_loader_2, replace_sampler=False, move_to_device=False)

        scheduler = solver_utils.build_lr_scheduler(cfg, optimizer, total_iters=max_iter)

        # resume or load model ===================================
        extra_ckpt_dict = dict(
            optimizer=optimizer,
            scheduler=scheduler,
        )
        if hasattr(self._precision_plugin, "scaler"):
            extra_ckpt_dict["gradscaler"] = self._precision_plugin.scaler

        checkpointer = MyCheckpointer(
            model,
            cfg.OUTPUT_DIR,
            save_to_disk=self.is_global_zero,
            **extra_ckpt_dict,
        )
        start_iter = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1

        if cfg.SOLVER.CHECKPOINT_BY_EPOCH:
            ckpt_period = cfg.SOLVER.CHECKPOINT_PERIOD * iters_per_epoch
        else:
            ckpt_period = cfg.SOLVER.CHECKPOINT_PERIOD
        periodic_checkpointer = PeriodicCheckpointer(
            checkpointer, ckpt_period, max_iter=max_iter, max_to_keep=cfg.SOLVER.MAX_TO_KEEP
        )

        # build writers ==============================================
        tbx_event_writer = self.get_tbx_event_writer(cfg.OUTPUT_DIR, backup=not cfg.get("RESUME", False))
        tbx_writer = tbx_event_writer._writer  # NOTE: we want to write some non-scalar data
        writers = (
            [MyCommonMetricPrinter(max_iter), MyJSONWriter(osp.join(cfg.OUTPUT_DIR, "metrics.json")), tbx_event_writer]
            if self.is_global_zero
            else []
        )

        # compared to "train_net.py", we do not support accurate timing and
        # precise BN here, because they are not trivial to implement
        logger.info("Starting training from iteration {}".format(start_iter))
        iter_time = None
        with EventStorage(start_iter) as storage:
            for iteration in range(start_iter, max_iter):
                storage.iter = iteration
                epoch = iteration // dataset_len + 1

                if np.random.rand() < train_2_ratio:
                    data = next(data_loader_2_iter)
                else:
                    data = next(data_loader_iter)

                if iter_time is not None:
                    storage.put_scalar("time", time.perf_counter() - iter_time)
                iter_time = time.perf_counter()

                # forward ============================================================
              
                batch = batch_data(cfg, data)
               
                out_dict, loss_dict = model(
                    batch["roi_img"],
                    gt_xyz=batch.get("roi_xyz", None),
                    gt_xyz_bin=batch.get("roi_xyz_bin", None),
                    gt_mask_trunc=batch["roi_mask_trunc"],
                    gt_mask_visib=batch["roi_mask_visib"],
                    gt_mask_obj=batch["roi_mask_obj"],
                    gt_region=batch.get("roi_region", None),
                    gt_allo_quat=batch.get("allo_quat", None),
                    gt_ego_quat=batch.get("ego_quat", None),
                    gt_allo_rot6d=batch.get("allo_rot6d", None),
                    gt_ego_rot6d=batch.get("ego_rot6d", None),
                    gt_ego_rot=batch.get("ego_rot", None),
                    gt_trans=batch.get("trans", None),
                    gt_trans_ratio=batch["roi_trans_ratio"],
                    gt_points=batch.get("roi_points", None),
                    sym_infos=batch.get("sym_info", None),
                    roi_classes=batch["roi_cls"],
                    roi_cams=batch["roi_cam"],
                    roi_whs=batch["roi_wh"],
                    roi_centers=batch["roi_center"],
                    resize_ratios=batch["resize_ratio"],
                    roi_coord_2d=batch.get("roi_coord_2d", None),
                    roi_extents=batch.get("roi_extent", None),
                    do_loss=True,
                )
                losses = sum(loss_dict.values())
                assert torch.isfinite(losses).all(), loss_dict

                loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                if self.is_global_zero:
                    storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

                optimizer.zero_grad(set_to_none=True)
                self.backward(losses)
                optimizer.step()

                storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
                scheduler.step()

                if (
                    cfg.TEST.EVAL_PERIOD > 0
                    and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0
                    and iteration != max_iter - 1
                ):
                    self.do_test(cfg, model, epoch=epoch, iteration=iteration)
                    # Compared to "train_net.py", the test results are not dumped to EventStorage
                    self.barrier()
                    torch.save(model, "best_current_model.pth")
                # iteration - start_iter > 5 and (
                    (iteration + 1) % cfg.TRAIN.PRINT_FREQ == 0 or iteration == max_iter - 1 or iteration < 100
                #)
              
                if True:
                    for writer in writers:
                        writer.write()
                    # visualize some images ========================================
                    if False:
                        with torch.no_grad():
                            vis_i = 0
                            roi_img_vis = batch["roi_img"][vis_i].cpu().numpy()
                            roi_img_vis = denormalize_image(roi_img_vis, cfg).transpose(1, 2, 0).astype("uint8")
                            tbx_writer.add_image("input_image", roi_img_vis, iteration)

                            out_coor_x = out_dict["coor_x"].detach()
                            out_coor_y = out_dict["coor_y"].detach()
                            out_coor_z = out_dict["coor_z"].detach()
                            out_xyz = get_out_coor(cfg, out_coor_x, out_coor_y, out_coor_z)

                            out_xyz_vis = out_xyz[vis_i].cpu().numpy().transpose(1, 2, 0)
                            out_xyz_vis = get_emb_show(out_xyz_vis)
                            tbx_writer.add_image("out_xyz", out_xyz_vis, iteration)

                            gt_xyz_vis = batch["roi_xyz"][vis_i].cpu().numpy().transpose(1, 2, 0)
                            gt_xyz_vis = get_emb_show(gt_xyz_vis)
                            tbx_writer.add_image("gt_xyz", gt_xyz_vis, iteration)

                            out_mask = out_dict["mask"].detach()
                            out_mask = get_out_mask(cfg, out_mask)
                            out_mask_vis = out_mask[vis_i, 0].cpu().numpy()
                            tbx_writer.add_image("out_mask", out_mask_vis, iteration)

                            gt_mask_vis = batch["roi_mask"][vis_i].detach().cpu().numpy()
                            tbx_writer.add_image("gt_mask", gt_mask_vis, iteration)

                if (iteration + 1) % periodic_checkpointer.period == 0 or (
                    periodic_checkpointer.max_iter is not None and (iteration + 1) >= periodic_checkpointer.max_iter
                ):
                    if hasattr(optimizer, "consolidate_state_dict"):  # for ddp_sharded
                        optimizer.consolidate_state_dict()
                periodic_checkpointer.step(iteration, epoch=epoch)

@loguru_logger.catch
def main(args):
    #1. Let's set up configs 
    cfg = setup(args)
    cfg.DATALOADER.NUM_WORKERS = 1 # decrease cpu usage here, now we need to modify the model later
    Lite(
        accelerator="gpu",
        devices=args.num_gpus,
        num_nodes=args.num_machines,
        precision=16 if cfg.SOLVER.AMP.ENABLED else 32,
    ).run(args, cfg)

if __name__ == "__main__":
    parser = my_default_argument_parser()    
    args = parser.parse_args()
    main(args)
