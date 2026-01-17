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

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, stage="fine"):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # means3D = pc.get_xyz
    # add deformation to each points
    # deformation = pc.get_deformation
    means3D = pc.get_xyz
    time_scalar = torch.tensor(viewpoint_camera.time, device=means3D.device)
    means2D = screenspace_points
    opacity = pc._opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc._scaling
        rotations = pc._rotation
    deformation_point = pc._deformation_table
    
    if stage == "coarse" :
        means3D_deform, scales_deform, rotations_deform, opacity_deform = means3D, scales, rotations, opacity
    else:
        if getattr(pc, "use_dct_deform", False) and getattr(pc, "_trajectory_coeffs", None) is not None:
            disp = pc.dct_displacement(time_scalar)
            mask_f = deformation_point.float().unsqueeze(-1)
            means3D_deform = means3D + disp * mask_f
            scales_deform = scales
            rotations_deform = rotations
            if getattr(pc, "dct_use_scale", False):
                ds = pc.dct_scale_delta(time_scalar)
                if ds is not None:
                    scales_deform = scales_deform + ds * mask_f
            if getattr(pc, "dct_use_rot", False):
                dr = pc.dct_rot_delta(time_scalar)
                if dr is not None:
                    rotations_deform = rotations_deform + dr * mask_f
            opacity_deform = opacity
        else:
            time = time_scalar.repeat(means3D.shape[0], 1)
            means3D_deform, scales_deform, rotations_deform, opacity_deform = pc._deformation(means3D[deformation_point], scales[deformation_point], 
                                                                             rotations[deformation_point], opacity[deformation_point],
                                                                             time[deformation_point])
    # print(time.max())
    with torch.no_grad():
        pc._deformation_accum[deformation_point] += torch.abs(means3D_deform - means3D[deformation_point])

    means3D_final = torch.zeros_like(means3D)
    rotations_final = torch.zeros_like(rotations)
    scales_final = torch.zeros_like(scales)
    opacity_final = torch.zeros_like(opacity)
    if means3D_deform is not None:
        means3D_deform = means3D_deform.to(means3D_final.dtype)
    if rotations_deform is not None:
        rotations_deform = rotations_deform.to(rotations_final.dtype)
    if scales_deform is not None:
        scales_deform = scales_deform.to(scales_final.dtype)
    if opacity_deform is not None:
        opacity_deform = opacity_deform.to(opacity_final.dtype)
    # Dense blend to avoid index_put_ overhead
    if deformation_point is not None:
        mask_f = deformation_point.float().unsqueeze(-1)
        means3D_final = means3D + (means3D_deform - means3D) * mask_f
        scales_final = scales + (scales_deform - scales) * mask_f
        rotations_final = rotations + (rotations_deform - rotations) * mask_f
        opacity_final = opacity + (opacity_deform - opacity) * mask_f
    else:
        means3D_final = means3D_deform
        scales_final = scales_deform
        rotations_final = rotations_deform
        opacity_final = opacity_deform

    scales_final = pc.scaling_activation(scales_final)
    rotations_final = pc.rotation_activation(rotations_final)
    opacity = pc.opacity_activation(opacity)

    # 预剔除：渲染前丢掉低贡献点（推理加速，不改变训练默认行为）
    means2D_final = means2D
    mask = None
    if getattr(pc, "pre_cull", False):
        op_thr = float(getattr(pc, "pre_cull_opacity", 0.0))
        min_r = float(getattr(pc, "pre_cull_min_radius", 0.0))
        if op_thr > 0.0:
            mask = opacity.squeeze(-1) >= op_thr
        else:
            mask = torch.ones_like(opacity.squeeze(-1), dtype=torch.bool)

        if min_r > 0.0:
            pts = means3D_final.float()
            ones = torch.ones((pts.shape[0], 1), device=pts.device, dtype=pts.dtype)
            pts_h = torch.cat([pts, ones], dim=1)
            view = viewpoint_camera.world_view_transform.to(pts.device, dtype=pts.dtype)
            pts_cam = pts_h @ view.T
            z = pts_cam[:, 2].abs().clamp_min(1e-6)
            fx = 0.5 * float(viewpoint_camera.image_width) / math.tan(float(viewpoint_camera.FoVx) * 0.5)
            max_scale = scales_final.max(dim=1).values.float()
            radius = max_scale * fx / z
            mask = mask & (radius >= min_r)

        if mask.any():
            means3D_final = means3D_final[mask]
            scales_final = scales_final[mask]
            rotations_final = rotations_final[mask]
            opacity = opacity[mask]
            means2D_final = means2D[mask]
        else:
            # 如果全被剔除，就退化为不剔除，避免空输入
            mask = None

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.cuda().repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    if mask is not None:
        if shs is not None:
            shs = shs[mask]
        if colors_precomp is not None:
            colors_precomp = colors_precomp[mask]

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    # Rasterizer expects float32 inputs; ensure fp16_static does not break dtype.
    def _fp32(x):
        return x.float() if torch.is_tensor(x) and x.dtype != torch.float32 else x

    means3D_final = _fp32(means3D_final)
    scales_final = _fp32(scales_final)
    rotations_final = _fp32(rotations_final)
    opacity = _fp32(opacity)
    cov3D_precomp = _fp32(cov3D_precomp)
    shs = _fp32(shs)
    colors_precomp = _fp32(colors_precomp)

    rendered_image, radii, depth = rasterizer(
        means3D = means3D_final,
        means2D = means2D_final,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales_final,
        rotations = rotations_final,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "depth": depth,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,}
