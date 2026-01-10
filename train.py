#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import numpy as np
import copy
import random
import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, lpips_loss, TV_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from torch.utils.data import DataLoader
from utils.timer import Timer
from torchmetrics.functional.regression import pearson_corrcoef

import lpips
from utils.scene_utils import render_training_image
from time import time
to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations, 
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, stage, tb_writer, train_iter, timer):
    first_iter = 0
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0
    
    final_iter = train_iter
    progress_bar = tqdm(range(first_iter, final_iter), desc="Training progress")
    first_iter += 1
    
    lpips_model = lpips.LPIPS(net="vgg").cuda()
    video_cams = scene.getVideoCameras()
    
    if not viewpoint_stack:
        viewpoint_stack = scene.getTrainCameras()
    
    for iteration in range(first_iter, final_iter+1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer, ts = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer, stage="stage")["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()
        gaussians.update_learning_rate(iteration)
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 500 == 0:
            gaussians.oneupSHdegree()
        if stage == 'coarse':
            idx = 0
        else:
            idx = randint(0, len(viewpoint_stack)-1)
        viewpoint_cams = [viewpoint_stack[idx]]

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
            
        images = []
        depths = []
        gt_images = []
        gt_depths = []
        masks = []
        
        radii_list = []
        visibility_filter_list = []
        viewspace_point_tensor_list = []
        
        for viewpoint_cam in viewpoint_cams:
            render_pkg = render(viewpoint_cam, gaussians, pipe, background, stage=stage)
            image, depth, viewspace_point_tensor, visibility_filter, radii = \
                render_pkg["render"], render_pkg["depth"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            gt_image = viewpoint_cam.original_image.cuda().float()
            gt_depth = viewpoint_cam.original_depth.cuda().float()
            mask = viewpoint_cam.mask.cuda()
            
            images.append(image.unsqueeze(0))
            depths.append(depth.unsqueeze(0))
            gt_images.append(gt_image.unsqueeze(0))
            gt_depths.append(gt_depth.unsqueeze(0))
            masks.append(mask.unsqueeze(0))
            radii_list.append(radii.unsqueeze(0))
            visibility_filter_list.append(visibility_filter.unsqueeze(0))
            viewspace_point_tensor_list.append(viewspace_point_tensor)
            
        radii = torch.cat(radii_list,0).max(dim=0).values
        visibility_filter = torch.cat(visibility_filter_list).any(dim=0)
        rendered_images = torch.cat(images,0)
        rendered_depths = torch.cat(depths, 0)
        gt_images = torch.cat(gt_images,0)
        gt_depths = torch.cat(gt_depths, 0)
        masks = torch.cat(masks, 0)
        
        Ll1 = l1_loss(rendered_images, gt_images, masks)
        
        if (gt_depths!=0).sum() < 10:
            depth_loss = torch.tensor(0.).cuda()
        elif scene.mode == 'binocular':
            rendered_depths[rendered_depths!=0] = 1 / rendered_depths[rendered_depths!=0]
            gt_depths[gt_depths!=0] = 1 / gt_depths[gt_depths!=0]
            depth_loss = l1_loss(rendered_depths, gt_depths, masks)
        elif scene.mode == 'monocular':
            rendered_depths_reshape = rendered_depths.reshape(-1, 1)
            gt_depths_reshape = gt_depths.reshape(-1, 1)
            mask_tmp = mask.reshape(-1)
            rendered_depths_reshape, gt_depths_reshape = rendered_depths_reshape[mask_tmp!=0, :], gt_depths_reshape[mask_tmp!=0, :]
            depth_loss =  0.001 * (1 - pearson_corrcoef(gt_depths_reshape, rendered_depths_reshape))
        else:
            raise ValueError(f"{scene.mode} is not implemented.")
        
        depth_tvloss = TV_loss(rendered_depths)
        img_tvloss = TV_loss(rendered_images)
        tv_loss = 0.03 * (img_tvloss + depth_tvloss)
        
        loss = Ll1 + depth_loss + tv_loss

        psnr_ = psnr(rendered_images, gt_images, masks).mean().double()        
        
        if stage == "fine" and hyper.time_smoothness_weight != 0:
            tv_loss = gaussians.compute_regulation(2e-2, 2e-2, 2e-2)
            loss += tv_loss
        if opt.lambda_dssim != 0:
            ssim_loss = ssim(rendered_images,gt_images)
            loss += opt.lambda_dssim * (1.0-ssim_loss)
        if opt.lambda_lpips !=0:
            lpipsloss = lpips_loss(rendered_images,gt_images,lpips_model)
            loss += opt.lambda_lpips * lpipsloss
        
        loss.backward()
        viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)
        for idx in range(0, len(viewspace_point_tensor_list)):
            viewspace_point_tensor_grad = viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_psnr_for_log = 0.4 * psnr_ + 0.6 * ema_psnr_for_log
            total_point = gaussians._xyz.shape[0]
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}",
                                          "psnr": f"{psnr_:.{2}f}",
                                          "point":f"{total_point}"})
                progress_bar.update(10)
            if iteration == train_iter:
                progress_bar.close()

            # Log and save
            timer.pause()
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, [pipe, background], stage)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration, stage)
            if dataset.render_process:
                if (iteration < 1000 and iteration % 10 == 1) \
                    or (iteration < 3000 and iteration % 50 == 1) \
                        or (iteration < 10000 and iteration %  100 == 1) \
                            or (iteration < 60000 and iteration % 100 ==1):
                    render_training_image(scene, gaussians, video_cams, render, pipe, background, stage, iteration-1,timer.get_elapsed_time())
            timer.start()
            
            # Densification
            if iteration < opt.densify_until_iter :
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)

                if stage == "coarse":
                    opacity_threshold = opt.opacity_threshold_coarse
                    densify_threshold = opt.densify_grad_threshold_coarse
                else:    
                    opacity_threshold = opt.opacity_threshold_fine_init - iteration*(opt.opacity_threshold_fine_init - opt.opacity_threshold_fine_after)/(opt.densify_until_iter)  
                    densify_threshold = opt.densify_grad_threshold_fine_init - iteration*(opt.densify_grad_threshold_fine_init - opt.densify_grad_threshold_after)/(opt.densify_until_iter )  

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 :
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)
                    
                if iteration > opt.pruning_from_iter and iteration % opt.pruning_interval == 0:
                    size_threshold = 40 if iteration > opt.opacity_reset_interval else None
                    gaussians.prune(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)
                    
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    print("reset opacity")
                    gaussians.reset_opacity()
                    
            # Optimizer step
            if iteration < train_iter:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def training(dataset, hyper, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, expname, extra_mark):
    tb_writer = prepare_output_and_logger(expname)
    gaussians = GaussianModel(dataset.sh_degree, hyper)
    dataset.model_path = args.model_path
    timer = Timer()
    scene = Scene(dataset, gaussians, load_coarse=args.no_fine)
    timer.start()
    scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                             checkpoint_iterations, checkpoint, debug_from,
                             gaussians, scene, "coarse", tb_writer, opt.coarse_iterations,timer)
    if not args.no_fine:
        scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                            checkpoint_iterations, checkpoint, debug_from,
                            gaussians, scene, "fine", tb_writer, opt.iterations,timer)

def save_gaussians_to(model_path, gaussians, iteration, stage):
    if stage == "coarse":
        iter_dir = os.path.join(model_path, "point_cloud", f"coarse_iteration_{iteration}")
    else:
        iter_dir = os.path.join(model_path, "point_cloud", f"iteration_{iteration}")
    os.makedirs(iter_dir, exist_ok=True)
    gaussians.save_ply(os.path.join(iter_dir, "point_cloud.ply"))
    gaussians.save_deformation(iter_dir)

def distill_dct_training(dataset, hyper, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, expname):
    tb_writer = prepare_output_and_logger(expname)
    if not getattr(hyper, "use_dct_deform", False):
        raise RuntimeError("distill_dct requires use_dct_deform=True in config or args.")

    teacher_model_path = args.teacher_model_path
    if not teacher_model_path:
        raise RuntimeError("--teacher_model_path is required for distill_dct")
    student_model_path = args.student_model_path or teacher_model_path

    cfg_path = os.path.join(teacher_model_path, "cfg_args")
    if os.path.exists(cfg_path):
        from argparse import Namespace
        cfg_txt = open(cfg_path, "r", encoding="utf-8", errors="ignore").read().strip()
        try:
            cfg_ns = eval(cfg_txt, {"Namespace": Namespace})
        except Exception:
            cfg_ns = None
        if cfg_ns is not None:
            for key in ("source_path", "extra_mark", "images", "mode", "white_background", "no_fine"):
                if hasattr(cfg_ns, key):
                    setattr(dataset, key, getattr(cfg_ns, key))
            if getattr(dataset, "source_path", None):
                sp = dataset.source_path
                if not os.path.isabs(sp):
                    dataset.source_path = os.path.abspath(sp)
            # rewrite cfg_args for student output so downstream tools can find data
            args.source_path = getattr(dataset, "source_path", "")
            args.extra_mark = getattr(dataset, "extra_mark", args.extra_mark)
            args.images = getattr(dataset, "images", args.images)
            args.mode = getattr(dataset, "mode", args.mode)
            args.white_background = getattr(dataset, "white_background", args.white_background)
            args.no_fine = getattr(dataset, "no_fine", args.no_fine)
            try:
                with open(os.path.join(args.model_path, "cfg_args"), "w", encoding="utf-8") as f:
                    f.write(str(Namespace(**vars(args))))
            except Exception:
                pass

    # Teacher scene (deformnet)
    teacher_dataset = copy.deepcopy(dataset)
    teacher_dataset.model_path = teacher_model_path
    teacher_hyper = copy.deepcopy(hyper)
    teacher_hyper.use_dct_deform = False
    teacher_gaussians = GaussianModel(dataset.sh_degree, teacher_hyper)
    teacher_scene = Scene(teacher_dataset, teacher_gaussians, load_iteration=args.distill_iteration, shuffle=False, load_coarse=args.no_fine)
    iteration = teacher_scene.loaded_iter

    # Student gaussians (DCT)
    student_gaussians = GaussianModel(dataset.sh_degree, hyper)
    iter_dir = os.path.join(student_model_path, "point_cloud", f"iteration_{iteration}")
    if not os.path.exists(iter_dir):
        iter_dir = os.path.join(student_model_path, "point_cloud", f"coarse_iteration_{iteration}")
    student_gaussians.load_ply(os.path.join(iter_dir, "point_cloud.ply"))
    student_gaussians.load_model(iter_dir)
    if student_gaussians.dct_use_codebook_pos:
        if student_gaussians._dct_codebook_pos is None or student_gaussians._dct_codebook_indices_pos is None:
            raise RuntimeError("DCT pos codebook requested but missing in dct_coeffs.pth")
    elif student_gaussians._trajectory_coeffs is None:
        student_gaussians._trajectory_coeffs = torch.nn.Parameter(
            torch.zeros((student_gaussians.get_xyz.shape[0], student_gaussians.dct_k, 3), device="cuda").requires_grad_(True)
        )
    if student_gaussians.dct_use_scale and student_gaussians._trajectory_coeffs_scale is None:
        student_gaussians._trajectory_coeffs_scale = torch.nn.Parameter(
            torch.zeros((student_gaussians.get_xyz.shape[0], student_gaussians.dct_k, 3), device="cuda").requires_grad_(True)
        )
    if student_gaussians.dct_use_rot and student_gaussians._trajectory_coeffs_rot is None:
        student_gaussians._trajectory_coeffs_rot = torch.nn.Parameter(
            torch.zeros((student_gaussians.get_xyz.shape[0], student_gaussians.dct_k, 4), device="cuda").requires_grad_(True)
        )
    if student_gaussians.dct_use_gate and student_gaussians._dct_log_alpha is None:
        student_gaussians._dct_log_alpha = torch.nn.Parameter(
            torch.full((student_gaussians.dct_k,), student_gaussians.dct_gate_init, device="cuda", dtype=torch.float32).requires_grad_(True)
        )

    # Freeze everything except DCT params
    for p in [student_gaussians._xyz, student_gaussians._features_dc, student_gaussians._features_rest,
              student_gaussians._opacity, student_gaussians._scaling, student_gaussians._rotation]:
        p.requires_grad_(False)
    for p in student_gaussians._deformation.parameters():
        p.requires_grad_(False)

    params = []
    if student_gaussians.dct_use_codebook_pos:
        params.append(student_gaussians._dct_codebook_pos)
        if student_gaussians._dct_codebook_residual_pos is not None:
            params.append(student_gaussians._dct_codebook_residual_pos)
    else:
        params.append(student_gaussians._trajectory_coeffs)
    if student_gaussians.dct_use_gate and student_gaussians._dct_log_alpha is not None:
        params.append(student_gaussians._dct_log_alpha)
    if student_gaussians._trajectory_coeffs_scale is not None:
        params.append(student_gaussians._trajectory_coeffs_scale)
    if student_gaussians._trajectory_coeffs_rot is not None:
        params.append(student_gaussians._trajectory_coeffs_rot)
    if student_gaussians.dct_use_codebook_scale and student_gaussians._dct_codebook_scale is not None:
        params.append(student_gaussians._dct_codebook_scale)
        if student_gaussians._dct_codebook_residual_scale is not None:
            params.append(student_gaussians._dct_codebook_residual_scale)
    if student_gaussians.dct_use_codebook_rot and student_gaussians._dct_codebook_rot is not None:
        params.append(student_gaussians._dct_codebook_rot)
        if student_gaussians._dct_codebook_residual_rot is not None:
            params.append(student_gaussians._dct_codebook_residual_rot)
    if args.dct_xyz_lr_mult > 0:
        student_gaussians._xyz.requires_grad_(True)
        params.append(student_gaussians._xyz)
    if args.distill_unfreeze_all:
        for p in [student_gaussians._features_dc, student_gaussians._features_rest,
                  student_gaussians._opacity, student_gaussians._scaling, student_gaussians._rotation]:
            p.requires_grad_(True)
            params.append(p)
    if student_gaussians.spatial_lr_scale == 0:
        student_gaussians.spatial_lr_scale = 1.0
    base_lr = opt.deformation_lr_init * student_gaussians.spatial_lr_scale
    lr_map = {
        id(student_gaussians._trajectory_coeffs): base_lr * args.dct_lr_mult,
        id(student_gaussians._dct_codebook_pos): base_lr * args.dct_lr_mult,
        id(student_gaussians._dct_codebook_residual_pos): base_lr * args.dct_lr_mult,
        id(student_gaussians._dct_codebook_scale): base_lr * args.dct_lr_mult,
        id(student_gaussians._dct_codebook_residual_scale): base_lr * args.dct_lr_mult,
        id(student_gaussians._dct_codebook_rot): base_lr * args.dct_lr_mult,
        id(student_gaussians._dct_codebook_residual_rot): base_lr * args.dct_lr_mult,
        id(student_gaussians._dct_log_alpha): base_lr * args.dct_lr_mult,
        id(student_gaussians._trajectory_coeffs_scale): base_lr * args.dct_lr_mult,
        id(student_gaussians._trajectory_coeffs_rot): base_lr * args.dct_lr_mult,
        id(student_gaussians._xyz): opt.position_lr_init * args.dct_xyz_lr_mult,
        id(student_gaussians._features_dc): opt.feature_lr * args.distill_unfreeze_lr_mult,
        id(student_gaussians._features_rest): (opt.feature_lr / 20.0) * args.distill_unfreeze_lr_mult,
        id(student_gaussians._opacity): opt.opacity_lr * args.distill_unfreeze_lr_mult,
        id(student_gaussians._scaling): opt.scaling_lr * args.distill_unfreeze_lr_mult,
        id(student_gaussians._rotation): opt.rotation_lr * args.distill_unfreeze_lr_mult,
    }
    param_groups = []
    for p in params:
        lr = lr_map.get(id(p), base_lr)
        param_groups.append({"params": [p], "lr": lr})
    optimizer = torch.optim.Adam(param_groups, eps=1e-15)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    viewpoint_stack = teacher_scene.getTrainCameras()

    ema_loss_for_log = 0.0
    lr_decay_steps = set(args.dct_lr_decay_iters)
    if not lr_decay_steps:
        lr_decay_steps = {int(args.distill_iterations * 0.6), int(args.distill_iterations * 0.8)}
    progress_bar = tqdm(range(1, args.distill_iterations + 1), desc="Distill DCT")
    for iteration in range(1, args.distill_iterations + 1):
        idx = randint(0, len(viewpoint_stack)-1)
        viewpoint_cam = viewpoint_stack[idx]

        with torch.inference_mode():
            teacher_pkg = render(viewpoint_cam, teacher_gaussians, pipe, background, stage="fine")
            teacher_img = teacher_pkg["render"].detach()

        student_pkg = render(viewpoint_cam, student_gaussians, pipe, background, stage="fine")
        student_img = student_pkg["render"]
        mask = viewpoint_cam.mask.cuda().float()

        ssim_w = max(0.0, min(1.0, args.distill_ssim_weight))
        l1_w = 1.0 - ssim_w
        l1 = l1_loss(student_img.unsqueeze(0), teacher_img.unsqueeze(0), mask.unsqueeze(0))
        if ssim_w > 0:
            mask3 = mask
            ssim_val = ssim((student_img * mask3).unsqueeze(0), (teacher_img * mask3).unsqueeze(0))
            loss = l1_w * l1 + ssim_w * (1.0 - ssim_val)
        else:
            loss = l1
        if student_gaussians.dct_use_gate and args.dct_gate_lambda > 0 and student_gaussians._dct_log_alpha is not None:
            loss = loss + args.dct_gate_lambda * torch.sigmoid(student_gaussians._dct_log_alpha).sum()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
        if iteration % 10 == 0:
            progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
            progress_bar.update(10)
        if iteration == args.distill_iterations:
            progress_bar.close()

        if iteration in lr_decay_steps:
            for group in optimizer.param_groups:
                group["lr"] *= args.dct_lr_decay_gamma

        if iteration in saving_iterations:
            print(f"\n[ITER {iteration}] Saving DCT Gaussians")
            save_gaussians_to(args.model_path, student_gaussians, iteration, "fine")

def prepare_output_and_logger(expname):    
    if not args.model_path:
        unique_str = expname
        args.model_path = os.path.join("./output/", unique_str)
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, stage):
    if tb_writer:
        tb_writer.add_scalar(f'{stage}/train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'{stage}/train_loss_patchestotal_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'{stage}/iter_time', elapsed, iteration)
    
    # Report test and samples of training set
    '''
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : [scene.getTestCameras()[idx % len(scene.getTestCameras())] for idx in range(10, 5000, 299)]},
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(10, 5000, 299)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians,stage=stage, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    mask = viewpoint.mask.to("cuda")
                    
                    image, gt_image, mask = image.unsqueeze(0), gt_image.unsqueeze(0), mask.unsqueeze(0)
                    
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(stage + "/"+config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(stage + "/"+config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image, mask).mean().double()
                    psnr_test += psnr(image, gt_image, mask).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(stage + "/"+config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(stage+"/"+config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram(f"{stage}/scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            
            tb_writer.add_scalar(f'{stage}/total_points', scene.gaussians.get_xyz.shape[0], iteration)
            tb_writer.add_scalar(f'{stage}/deformation_rate', scene.gaussians._deformation_table.sum()/scene.gaussians.get_xyz.shape[0], iteration)
            tb_writer.add_histogram(f"{stage}/scene/motion_histogram", scene.gaussians._deformation_accum.mean(dim=-1)/100, iteration,max_bins=500)
        
        torch.cuda.empty_cache()
        '''

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    # Set up command line argument parser
    # torch.set_default_tensor_type('torch.FloatTensor')
    torch.cuda.empty_cache()
    parser = ArgumentParser(description="Training script parameters")
    setup_seed(6666)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[i*500 for i in range(0,120)])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[2000, 3000,4000, 5000, 6000, 9000, 10000, 14000, 20000, 30_000,45000,60000])
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--expname", type=str, default = "")
    parser.add_argument("--configs", type=str, default = "")
    parser.add_argument("--distill_dct", action="store_true", default=False)
    parser.add_argument("--teacher_model_path", type=str, default=None)
    parser.add_argument("--student_model_path", type=str, default=None)
    parser.add_argument("--distill_iteration", type=int, default=-1)
    parser.add_argument("--distill_iterations", type=int, default=2000)
    parser.add_argument("--dct_gate_lambda", type=float, default=0.0)
    parser.add_argument("--dct_lr_mult", type=float, default=10.0)
    parser.add_argument("--dct_xyz_lr_mult", type=float, default=0.0)
    parser.add_argument("--distill_ssim_weight", type=float, default=0.2)
    parser.add_argument("--dct_lr_decay_iters", nargs="+", type=int, default=[])
    parser.add_argument("--dct_lr_decay_gamma", type=float, default=0.3)
    parser.add_argument("--distill_unfreeze_all", action="store_true", default=False)
    parser.add_argument("--distill_unfreeze_lr_mult", type=float, default=0.01)
    parser.add_argument("--dct_use_codebook_pos", action="store_true", default=False)
    parser.add_argument("--dct_use_codebook_scale", action="store_true", default=False)
    parser.add_argument("--dct_use_codebook_rot", action="store_true", default=False)
    parser.add_argument("--dct_codebook_size_pos", type=int, default=256)
    parser.add_argument("--dct_codebook_size_scale", type=int, default=256)
    parser.add_argument("--dct_codebook_size_rot", type=int, default=256)
    parser.add_argument("--dct_expand_codebook", action="store_true", default=False)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    if args.distill_dct:
        distill_dct_training(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args),
                             args.test_iterations, args.save_iterations, args.checkpoint_iterations,
                             args.start_checkpoint, args.debug_from, args.expname)
    else:
        training(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, \
            args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.expname, args.extra_mark)

    # All done
    print("\nTraining complete.")
