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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from random import randint
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scene.deformation import deform_network
from scene.regulation import compute_plane_smoothness


class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree : int, args):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        
        self._deformation = deform_network(args)
        self.use_dct_deform = getattr(args, "use_dct_deform", False)
        self.dct_k = int(getattr(args, "dct_k", 8))
        self.dct_T = int(getattr(args, "dct_T", 200))
        self.dct_use_gate = getattr(args, "dct_use_gate", True)
        self.dct_gate_init = float(getattr(args, "dct_gate_init", 0.0))
        self.dct_use_scale = getattr(args, "dct_use_scale", False)
        self.dct_use_rot = getattr(args, "dct_use_rot", False)
        self.dct_use_codebook_pos = getattr(args, "dct_use_codebook_pos", False)
        self.dct_use_codebook_scale = getattr(args, "dct_use_codebook_scale", False)
        self.dct_use_codebook_rot = getattr(args, "dct_use_codebook_rot", False)
        self.dct_codebook_size_pos = int(getattr(args, "dct_codebook_size_pos", 256))
        self.dct_codebook_size_scale = int(getattr(args, "dct_codebook_size_scale", 256))
        self.dct_codebook_size_rot = int(getattr(args, "dct_codebook_size_rot", 256))
        self.dct_expand_codebook = getattr(args, "dct_expand_codebook", False)
        self.fp16_static = getattr(args, "fp16_static", False)
        self.dct_masked = getattr(args, "dct_masked", False)
        self.use_anchor_dct = getattr(args, "use_anchor_dct", False)
        self.anchor_k = int(getattr(args, "anchor_k", 4))
        self.anchor_residual_k = int(getattr(args, "anchor_residual_k", 0))
        self.anchor_residual_tail = bool(getattr(args, "anchor_residual_tail", True))
        self._dct_basis = None
        self._trajectory_coeffs = None
        self._trajectory_coeffs_scale = None
        self._trajectory_coeffs_rot = None
        self._anchor_positions = None
        self._anchor_coeffs = None
        self._anchor_indices = None
        self._anchor_weights = None
        self._anchor_residual_coeffs = None
        self._dct_log_alpha = None
        self._dct_codebook_pos = None
        self._dct_codebook_scale = None
        self._dct_codebook_rot = None
        self._dct_codebook_indices_pos = None
        self._dct_codebook_indices_scale = None
        self._dct_codebook_indices_rot = None
        self._dct_codebook_residual_pos = None
        self._dct_codebook_residual_scale = None
        self._dct_codebook_residual_rot = None

        self._deformation_table = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._deformation.state_dict(),
            self._deformation_table,
            self._trajectory_coeffs,
            self._trajectory_coeffs_scale,
            self._trajectory_coeffs_rot,
            self._dct_log_alpha,
            # self.grid,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.percent_dense,
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
            self._xyz, 
            self._deformation_table,
            self._deformation,
            self._trajectory_coeffs,
            self._trajectory_coeffs_scale,
            self._trajectory_coeffs_rot,
            self._dct_log_alpha,
            # self.grid,
            self._features_dc, 
            self._features_rest,
            self._scaling, 
            self._rotation, 
            self._opacity,
            self.max_radii2D, 
            xyz_gradient_accum, 
            denom,
            opt_dict, 
            self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, time_line: int):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._deformation = self._deformation.to("cuda") 
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self._deformation_table = torch.gt(torch.ones((self.get_xyz.shape[0]),device="cuda"),0)
        if self.use_dct_deform:
            if not self.use_anchor_dct:
                self._trajectory_coeffs = nn.Parameter(
                    torch.zeros((self.get_xyz.shape[0], self.dct_k, 3), device="cuda").requires_grad_(True)
                )
            if self.dct_use_scale:
                self._trajectory_coeffs_scale = nn.Parameter(
                    torch.zeros((self.get_xyz.shape[0], self.dct_k, 3), device="cuda").requires_grad_(True)
                )
            if self.dct_use_rot:
                self._trajectory_coeffs_rot = nn.Parameter(
                    torch.zeros((self.get_xyz.shape[0], self.dct_k, 4), device="cuda").requires_grad_(True)
                )
            if self.dct_use_gate:
                self._dct_log_alpha = nn.Parameter(
                    torch.full((self.dct_k,), self.dct_gate_init, device="cuda", dtype=torch.float32).requires_grad_(True)
                )
        self._apply_fp16_static()

    def _apply_fp16_static(self):
        if not self.fp16_static:
            return
        for name in ("_xyz", "_opacity", "_rotation"):
            t = getattr(self, name, None)
            if t is None or not torch.is_tensor(t):
                continue
            if isinstance(t, nn.Parameter):
                setattr(self, name, nn.Parameter(t.data.half(), requires_grad=t.requires_grad))
            else:
                setattr(self, name, t.half())
    
    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self._deformation_accum = torch.zeros((self.get_xyz.shape[0],3),device="cuda")
        
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': list(self._deformation.get_mlp_parameters()), 'lr': training_args.deformation_lr_init * self.spatial_lr_scale, "name": "deformation"},
            {'params': list(self._deformation.get_grid_parameters()), 'lr': training_args.grid_lr_init * self.spatial_lr_scale, "name": "grid"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]
        if self.use_dct_deform:
            if self.use_anchor_dct and self._anchor_coeffs is not None:
                l.append({'params': [self._anchor_coeffs], 'lr': training_args.deformation_lr_init * self.spatial_lr_scale, "name": "anchor_dct_coeffs"})
                if self._anchor_residual_coeffs is not None:
                    l.append({'params': [self._anchor_residual_coeffs], 'lr': training_args.deformation_lr_init * self.spatial_lr_scale, "name": "anchor_dct_residuals"})
            elif self._trajectory_coeffs is not None:
                l.append({'params': [self._trajectory_coeffs], 'lr': training_args.deformation_lr_init * self.spatial_lr_scale, "name": "dct_coeffs"})
            if self._trajectory_coeffs_scale is not None:
                l.append({'params': [self._trajectory_coeffs_scale], 'lr': training_args.deformation_lr_init * self.spatial_lr_scale, "name": "dct_coeffs_scale"})
            if self._trajectory_coeffs_rot is not None:
                l.append({'params': [self._trajectory_coeffs_rot], 'lr': training_args.deformation_lr_init * self.spatial_lr_scale, "name": "dct_coeffs_rot"})
            if self.dct_use_gate and self._dct_log_alpha is not None:
                l.append({'params': [self._dct_log_alpha], 'lr': training_args.deformation_lr_init * self.spatial_lr_scale, "name": "dct_gate"})
        if self.use_dct_deform and self.dct_use_codebook_pos and self._dct_codebook_pos is not None:
            l.append({'params': [self._dct_codebook_pos], 'lr': training_args.deformation_lr_init * self.spatial_lr_scale, "name": "dct_codebook_pos"})
            if self._dct_codebook_residual_pos is not None:
                l.append({'params': [self._dct_codebook_residual_pos], 'lr': training_args.deformation_lr_init * self.spatial_lr_scale, "name": "dct_residual_pos"})
        if self.use_dct_deform and self.dct_use_codebook_scale and self._dct_codebook_scale is not None:
            l.append({'params': [self._dct_codebook_scale], 'lr': training_args.deformation_lr_init * self.spatial_lr_scale, "name": "dct_codebook_scale"})
            if self._dct_codebook_residual_scale is not None:
                l.append({'params': [self._dct_codebook_residual_scale], 'lr': training_args.deformation_lr_init * self.spatial_lr_scale, "name": "dct_residual_scale"})
        if self.use_dct_deform and self.dct_use_codebook_rot and self._dct_codebook_rot is not None:
            l.append({'params': [self._dct_codebook_rot], 'lr': training_args.deformation_lr_init * self.spatial_lr_scale, "name": "dct_codebook_rot"})
            if self._dct_codebook_residual_rot is not None:
                l.append({'params': [self._dct_codebook_residual_rot], 'lr': training_args.deformation_lr_init * self.spatial_lr_scale, "name": "dct_residual_rot"})

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.deformation_scheduler_args = get_expon_lr_func(lr_init=training_args.deformation_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.deformation_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.deformation_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps) 
        self.grid_scheduler_args = get_expon_lr_func(lr_init=training_args.grid_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.grid_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.deformation_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)    

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                # return lr
            if  "grid" in param_group["name"]:
                lr = self.grid_scheduler_args(iteration)
                param_group['lr'] = lr
                # return lr
            elif param_group["name"] == "deformation":
                lr = self.deformation_scheduler_args(iteration)
                param_group['lr'] = lr
                # return lr
            elif param_group["name"] in ("dct_coeffs", "dct_gate"):
                lr = self.deformation_scheduler_args(iteration)
                param_group['lr'] = lr
            elif param_group["name"] == "anchor_dct_coeffs":
                lr = self.deformation_scheduler_args(iteration)
                param_group['lr'] = lr
            elif param_group["name"] == "anchor_dct_residuals":
                lr = self.deformation_scheduler_args(iteration)
                param_group['lr'] = lr
            elif param_group["name"] in ("dct_coeffs_scale", "dct_coeffs_rot"):
                lr = self.deformation_scheduler_args(iteration)
                param_group['lr'] = lr
            elif param_group["name"] in ("dct_codebook_pos", "dct_residual_pos", "dct_codebook_scale", "dct_residual_scale", "dct_codebook_rot", "dct_residual_rot"):
                lr = self.deformation_scheduler_args(iteration)
                param_group['lr'] = lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l
    
    def compute_deformation(self,time):    
        deform = self._deformation[:,:,:time].sum(dim=-1)
        xyz = self._xyz + deform
        return xyz

    def _ensure_dct_basis(self, device):
        if self._dct_basis is not None and self._dct_basis.device == device:
            return self._dct_basis
        T = max(1, int(self.dct_T))
        K = max(1, int(self.dct_k))
        t = torch.arange(T, device=device, dtype=torch.float32)
        k = torch.arange(K, device=device, dtype=torch.float32).unsqueeze(1)
        w = torch.full((K, 1), math.sqrt(2.0 / T), device=device, dtype=torch.float32)
        w[0, 0] = math.sqrt(1.0 / T)
        basis = w * torch.cos(math.pi * k * (2.0 * t + 1.0) / (2.0 * T))
        self._dct_basis = basis
        return basis

    def dct_gate(self):
        if self._dct_log_alpha is None:
            return None
        return torch.sigmoid(self._dct_log_alpha)

    def _select_dct_basis(self, time_tensor):
        basis = self._ensure_dct_basis(time_tensor.device)
        t = time_tensor.detach().float()
        if t.dim() > 1:
            t = t[:, 0]
        if t.numel() > 0 and float(t.max()) > 1.0:
            denom = max(1.0, float(self.dct_T - 1))
            t = t / denom
        t = torch.clamp(t, 0.0, 1.0)
        idx = torch.round(t * (basis.shape[1] - 1)).long().clamp(0, basis.shape[1] - 1)
        if idx.dim() == 0 or idx.numel() == 1:
            basis_sel = basis[:, idx].unsqueeze(1)  # [K, 1]
        else:
            basis_sel = basis[:, idx]  # [K, N]
        if self.dct_use_gate:
            gate = self.dct_gate()
            if gate is not None:
                basis_sel = basis_sel * gate[:, None]
        return basis_sel.transpose(0, 1)  # [N, K]

    def _get_dct_coeffs(self, kind):
        if kind == "pos":
            if self.dct_use_codebook_pos and self._dct_codebook_pos is not None and self._dct_codebook_indices_pos is not None:
                idx = self._dct_codebook_indices_pos.long()
                code = self._dct_codebook_pos[idx]
                if self._dct_codebook_residual_pos is not None:
                    return code + self._dct_codebook_residual_pos
                return code
            return self._trajectory_coeffs
        if kind == "scale":
            if self.dct_use_codebook_scale and self._dct_codebook_scale is not None and self._dct_codebook_indices_scale is not None:
                idx = self._dct_codebook_indices_scale.long()
                code = self._dct_codebook_scale[idx]
                if self._dct_codebook_residual_scale is not None:
                    return code + self._dct_codebook_residual_scale
                return code
            return self._trajectory_coeffs_scale
        if kind == "rot":
            if self.dct_use_codebook_rot and self._dct_codebook_rot is not None and self._dct_codebook_indices_rot is not None:
                idx = self._dct_codebook_indices_rot.long()
                code = self._dct_codebook_rot[idx]
                if self._dct_codebook_residual_rot is not None:
                    return code + self._dct_codebook_residual_rot
                return code
            return self._trajectory_coeffs_rot
        return None

    def _get_dct_coeffs_for_indices(self, kind, idx):
        if idx is None or idx.numel() == 0:
            return None
        if kind == "pos":
            if self.dct_use_codebook_pos and self._dct_codebook_pos is not None and self._dct_codebook_indices_pos is not None:
                cb_idx = self._dct_codebook_indices_pos[idx].long()
                code = self._dct_codebook_pos[cb_idx]
                if self._dct_codebook_residual_pos is not None:
                    return code + self._dct_codebook_residual_pos[idx]
                return code
            return self._trajectory_coeffs[idx] if self._trajectory_coeffs is not None else None
        if kind == "scale":
            if self.dct_use_codebook_scale and self._dct_codebook_scale is not None and self._dct_codebook_indices_scale is not None:
                cb_idx = self._dct_codebook_indices_scale[idx].long()
                code = self._dct_codebook_scale[cb_idx]
                if self._dct_codebook_residual_scale is not None:
                    return code + self._dct_codebook_residual_scale[idx]
                return code
            return self._trajectory_coeffs_scale[idx] if self._trajectory_coeffs_scale is not None else None
        if kind == "rot":
            if self.dct_use_codebook_rot and self._dct_codebook_rot is not None and self._dct_codebook_indices_rot is not None:
                cb_idx = self._dct_codebook_indices_rot[idx].long()
                code = self._dct_codebook_rot[cb_idx]
                if self._dct_codebook_residual_rot is not None:
                    return code + self._dct_codebook_residual_rot[idx]
                return code
            return self._trajectory_coeffs_rot[idx] if self._trajectory_coeffs_rot is not None else None
        return None

    def dct_displacement(self, time_tensor):
        if self.use_anchor_dct and self._anchor_coeffs is not None and self._anchor_indices is not None and self._anchor_weights is not None:
            basis_sel = self._select_dct_basis(time_tensor)
            if basis_sel.dim() == 1:
                basis_sel = basis_sel.unsqueeze(0)
            anchor_disp = torch.einsum("akd,nk->nad", self._anchor_coeffs, basis_sel).squeeze(0)
            idx = self._anchor_indices.long()
            w = self._anchor_weights.to(anchor_disp)
            disp = (anchor_disp[idx] * w.unsqueeze(-1)).sum(dim=1)
            if self._anchor_residual_coeffs is not None and self.anchor_residual_k > 0:
                if self.anchor_residual_tail:
                    basis_res = basis_sel[:, -self.anchor_residual_k:]
                else:
                    basis_res = basis_sel[:, :self.anchor_residual_k]
                disp = disp + torch.einsum("nkd,nk->nd", self._anchor_residual_coeffs, basis_res)
            return disp
        coeffs = self._get_dct_coeffs("pos")
        if coeffs is None:
            return None
        basis_sel = self._select_dct_basis(time_tensor)
        if basis_sel.shape[0] == 1 and coeffs.shape[0] != 1:
            basis_sel = basis_sel.expand(coeffs.shape[0], -1)
        return torch.einsum("nkd,nk->nd", coeffs, basis_sel)

    def dct_displacement_masked(self, time_tensor, idx):
        if self.use_anchor_dct and self._anchor_coeffs is not None and self._anchor_indices is not None and self._anchor_weights is not None:
            disp = self.dct_displacement(time_tensor)
            if disp is None:
                return None
            return disp[idx]
        coeffs = self._get_dct_coeffs_for_indices("pos", idx)
        if coeffs is None:
            return None
        basis_sel = self._select_dct_basis(time_tensor)
        if basis_sel.shape[0] == 1 and coeffs.shape[0] != 1:
            basis_sel = basis_sel.expand(coeffs.shape[0], -1)
        return torch.einsum("nkd,nk->nd", coeffs, basis_sel)

    def dct_scale_delta(self, time_tensor):
        coeffs = self._get_dct_coeffs("scale")
        if coeffs is None:
            return None
        basis_sel = self._select_dct_basis(time_tensor)
        if basis_sel.shape[0] == 1 and coeffs.shape[0] != 1:
            basis_sel = basis_sel.expand(coeffs.shape[0], -1)
        return torch.einsum("nkd,nk->nd", coeffs, basis_sel)

    def dct_scale_delta_masked(self, time_tensor, idx):
        coeffs = self._get_dct_coeffs_for_indices("scale", idx)
        if coeffs is None:
            return None
        basis_sel = self._select_dct_basis(time_tensor)
        if basis_sel.shape[0] == 1 and coeffs.shape[0] != 1:
            basis_sel = basis_sel.expand(coeffs.shape[0], -1)
        return torch.einsum("nkd,nk->nd", coeffs, basis_sel)

    def dct_rot_delta(self, time_tensor):
        coeffs = self._get_dct_coeffs("rot")
        if coeffs is None:
            return None
        basis_sel = self._select_dct_basis(time_tensor)
        if basis_sel.shape[0] == 1 and coeffs.shape[0] != 1:
            basis_sel = basis_sel.expand(coeffs.shape[0], -1)
        return torch.einsum("nkd,nk->nd", coeffs, basis_sel)

    def dct_rot_delta_masked(self, time_tensor, idx):
        coeffs = self._get_dct_coeffs_for_indices("rot", idx)
        if coeffs is None:
            return None
        basis_sel = self._select_dct_basis(time_tensor)
        if basis_sel.shape[0] == 1 and coeffs.shape[0] != 1:
            basis_sel = basis_sel.expand(coeffs.shape[0], -1)
        return torch.einsum("nkd,nk->nd", coeffs, basis_sel)

    def _expand_codebook(self, kind):
        if kind == "pos":
            if not (self.dct_use_codebook_pos and self._dct_codebook_pos is not None and self._dct_codebook_indices_pos is not None):
                return
            idx = self._dct_codebook_indices_pos.long()
            coeffs = self._dct_codebook_pos[idx]
            if self._dct_codebook_residual_pos is not None:
                coeffs = coeffs + self._dct_codebook_residual_pos
            self._trajectory_coeffs = nn.Parameter(coeffs.requires_grad_(True))
            self.dct_use_codebook_pos = False
            self._dct_codebook_pos = None
            self._dct_codebook_indices_pos = None
            self._dct_codebook_residual_pos = None
        elif kind == "scale":
            if not (self.dct_use_codebook_scale and self._dct_codebook_scale is not None and self._dct_codebook_indices_scale is not None):
                return
            idx = self._dct_codebook_indices_scale.long()
            coeffs = self._dct_codebook_scale[idx]
            if self._dct_codebook_residual_scale is not None:
                coeffs = coeffs + self._dct_codebook_residual_scale
            self._trajectory_coeffs_scale = nn.Parameter(coeffs.requires_grad_(True))
            self.dct_use_codebook_scale = False
            self._dct_codebook_scale = None
            self._dct_codebook_indices_scale = None
            self._dct_codebook_residual_scale = None
        elif kind == "rot":
            if not (self.dct_use_codebook_rot and self._dct_codebook_rot is not None and self._dct_codebook_indices_rot is not None):
                return
            idx = self._dct_codebook_indices_rot.long()
            coeffs = self._dct_codebook_rot[idx]
            if self._dct_codebook_residual_rot is not None:
                coeffs = coeffs + self._dct_codebook_residual_rot
            self._trajectory_coeffs_rot = nn.Parameter(coeffs.requires_grad_(True))
            self.dct_use_codebook_rot = False
            self._dct_codebook_rot = None
            self._dct_codebook_indices_rot = None
            self._dct_codebook_residual_rot = None

    def load_model(self, path):
        print("loading model from exists{}".format(path))
        
        weight_dict = torch.load(os.path.join(path,"deformation.pth"),map_location="cuda")
        self._deformation.load_state_dict(weight_dict)
        self._deformation = self._deformation.to("cuda")
        
        self._deformation_table = torch.gt(torch.ones((self.get_xyz.shape[0]),device="cuda"),0)
        if os.path.exists(os.path.join(path, "deformation_table.pth")):
            self._deformation_table = torch.load(os.path.join(path, "deformation_table.pth"),map_location="cuda")
            
        self._deformation_accum = torch.zeros((self.get_xyz.shape[0],3),device="cuda")
        if os.path.exists(os.path.join(path, "deformation_accum.pth")):
            self._deformation_accum = torch.load(os.path.join(path, "deformation_accum.pth"),map_location="cuda")
        
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        dct_path = os.path.join(path, "dct_coeffs.pth")
        if not os.path.exists(dct_path):
            xz_path = dct_path + ".xz"
            if os.path.exists(xz_path):
                import lzma
                import pickle
                data = lzma.decompress(open(xz_path, "rb").read())
                dct = pickle.loads(data)
            else:
                dct = None
        else:
            dct = torch.load(dct_path, map_location="cuda")
        if dct is not None:
            self.use_dct_deform = True
            self.dct_k = int(dct.get("dct_k", self.dct_k))
            self.dct_T = int(dct.get("dct_T", self.dct_T))
            codebook_pos = dct.get("codebook_pos", None)
            indices_pos = dct.get("indices_pos", None)
            residuals_pos = dct.get("residuals_pos", None)
            residuals_pos_int8 = dct.get("residuals_pos_int8", None)
            residuals_pos_scale = dct.get("residuals_pos_scale", None)
            if codebook_pos is not None and indices_pos is not None:
                self.dct_use_codebook_pos = True
                self._dct_codebook_pos = nn.Parameter(codebook_pos.to("cuda").float().requires_grad_(True))
                self._dct_codebook_indices_pos = indices_pos.to("cuda")
                if residuals_pos_int8 is not None and residuals_pos_scale is not None:
                    res = residuals_pos_int8.to("cuda").float() * float(residuals_pos_scale)
                    self._dct_codebook_residual_pos = nn.Parameter(res.requires_grad_(True))
                elif residuals_pos is not None:
                    self._dct_codebook_residual_pos = nn.Parameter(residuals_pos.to("cuda").float().requires_grad_(True))
            else:
                coeffs = dct.get("coeffs", None)
                if coeffs is not None:
                    self._trajectory_coeffs = nn.Parameter(coeffs.to("cuda").requires_grad_(True))
            coeffs_scale = dct.get("coeffs_scale", None)
            if coeffs_scale is not None:
                self._trajectory_coeffs_scale = nn.Parameter(coeffs_scale.to("cuda").requires_grad_(True))
                self.dct_use_scale = True
            codebook_scale = dct.get("codebook_scale", None)
            indices_scale = dct.get("indices_scale", None)
            residuals_scale = dct.get("residuals_scale", None)
            residuals_scale_int8 = dct.get("residuals_scale_int8", None)
            residuals_scale_scale = dct.get("residuals_scale_scale", None)
            if codebook_scale is not None and indices_scale is not None:
                self.dct_use_codebook_scale = True
                self.dct_use_scale = True
                self._dct_codebook_scale = nn.Parameter(codebook_scale.to("cuda").float().requires_grad_(True))
                self._dct_codebook_indices_scale = indices_scale.to("cuda")
                if residuals_scale_int8 is not None and residuals_scale_scale is not None:
                    res = residuals_scale_int8.to("cuda").float() * float(residuals_scale_scale)
                    self._dct_codebook_residual_scale = nn.Parameter(res.requires_grad_(True))
                elif residuals_scale is not None:
                    self._dct_codebook_residual_scale = nn.Parameter(residuals_scale.to("cuda").float().requires_grad_(True))
            coeffs_rot = dct.get("coeffs_rot", None)
            if coeffs_rot is not None:
                self._trajectory_coeffs_rot = nn.Parameter(coeffs_rot.to("cuda").requires_grad_(True))
                self.dct_use_rot = True
            codebook_rot = dct.get("codebook_rot", None)
            indices_rot = dct.get("indices_rot", None)
            residuals_rot = dct.get("residuals_rot", None)
            residuals_rot_int8 = dct.get("residuals_rot_int8", None)
            residuals_rot_scale = dct.get("residuals_rot_scale", None)
            if codebook_rot is not None and indices_rot is not None:
                self.dct_use_codebook_rot = True
                self.dct_use_rot = True
                self._dct_codebook_rot = nn.Parameter(codebook_rot.to("cuda").float().requires_grad_(True))
                self._dct_codebook_indices_rot = indices_rot.to("cuda")
                if residuals_rot_int8 is not None and residuals_rot_scale is not None:
                    res = residuals_rot_int8.to("cuda").float() * float(residuals_rot_scale)
                    self._dct_codebook_residual_rot = nn.Parameter(res.requires_grad_(True))
                elif residuals_rot is not None:
                    self._dct_codebook_residual_rot = nn.Parameter(residuals_rot.to("cuda").float().requires_grad_(True))
            log_alpha = dct.get("log_alpha", None)
            if log_alpha is not None:
                self._dct_log_alpha = nn.Parameter(log_alpha.to("cuda").requires_grad_(True))
            if self.dct_expand_codebook:
                with torch.no_grad():
                    self._expand_codebook("pos")
                    self._expand_codebook("scale")
                    self._expand_codebook("rot")
        if self.use_anchor_dct:
            anchor_path = os.path.join(path, "anchor_dct.pth")
            if os.path.exists(anchor_path):
                anchor = torch.load(anchor_path, map_location="cuda")
                anchor_pos = anchor.get("anchor_pos", None)
                if anchor_pos is not None:
                    self._anchor_positions = anchor_pos.to("cuda").float()
                coeffs = anchor.get("anchor_coeffs", None)
                if coeffs is not None:
                    self._anchor_coeffs = nn.Parameter(coeffs.to("cuda").float().requires_grad_(True))
                anchor_idx = anchor.get("anchor_idx", None)
                if anchor_idx is not None:
                    self._anchor_indices = anchor_idx.to("cuda").long()
                anchor_w = anchor.get("anchor_w", None)
                if anchor_w is not None:
                    self._anchor_weights = anchor_w.to("cuda").float()
                self.anchor_k = int(anchor.get("anchor_k", self.anchor_k))
                self.anchor_residual_k = int(anchor.get("residual_k", self.anchor_residual_k))
                self.anchor_residual_tail = bool(anchor.get("residual_tail", self.anchor_residual_tail))
                residuals = anchor.get("residual_coeffs", None)
                if residuals is not None:
                    self._anchor_residual_coeffs = nn.Parameter(residuals.to("cuda").float().requires_grad_(True))
                self.use_dct_deform = True
            else:
                raise FileNotFoundError(f"Missing anchor_dct.pth in {path} while use_anchor_dct=True")

    def save_deformation(self, path):
        torch.save(self._deformation.state_dict(),os.path.join(path, "deformation.pth"))
        torch.save(self._deformation_table,os.path.join(path, "deformation_table.pth"))
        torch.save(self._deformation_accum,os.path.join(path, "deformation_accum.pth"))
        if self.use_dct_deform:
            dct = {
                "coeffs": self._trajectory_coeffs.detach().cpu() if self._trajectory_coeffs is not None else None,
                "coeffs_scale": self._trajectory_coeffs_scale.detach().cpu() if self._trajectory_coeffs_scale is not None else None,
                "coeffs_rot": self._trajectory_coeffs_rot.detach().cpu() if self._trajectory_coeffs_rot is not None else None,
                "log_alpha": self._dct_log_alpha.detach().cpu() if self._dct_log_alpha is not None else None,
                "dct_k": self.dct_k,
                "dct_T": self.dct_T,
            }
            if self.dct_use_codebook_pos and self._dct_codebook_pos is not None and self._dct_codebook_indices_pos is not None:
                dct["codebook_pos"] = self._dct_codebook_pos.detach().cpu()
                dct["indices_pos"] = self._dct_codebook_indices_pos.detach().to(torch.uint8).cpu()
                dct["residuals_pos"] = self._dct_codebook_residual_pos.detach().to(torch.float16).cpu() if self._dct_codebook_residual_pos is not None else None
                dct["coeffs"] = None
            if self.dct_use_codebook_scale and self._dct_codebook_scale is not None and self._dct_codebook_indices_scale is not None:
                dct["codebook_scale"] = self._dct_codebook_scale.detach().cpu()
                dct["indices_scale"] = self._dct_codebook_indices_scale.detach().to(torch.uint8).cpu()
                dct["residuals_scale"] = self._dct_codebook_residual_scale.detach().to(torch.float16).cpu() if self._dct_codebook_residual_scale is not None else None
                dct["coeffs_scale"] = None
            if self.dct_use_codebook_rot and self._dct_codebook_rot is not None and self._dct_codebook_indices_rot is not None:
                dct["codebook_rot"] = self._dct_codebook_rot.detach().cpu()
                dct["indices_rot"] = self._dct_codebook_indices_rot.detach().to(torch.uint8).cpu()
                dct["residuals_rot"] = self._dct_codebook_residual_rot.detach().to(torch.float16).cpu() if self._dct_codebook_residual_rot is not None else None
                dct["coeffs_rot"] = None
            torch.save(dct, os.path.join(path, "dct_coeffs.pth"))
        if self.use_anchor_dct and self._anchor_coeffs is not None and self._anchor_indices is not None and self._anchor_weights is not None:
            anchor = {
                "anchor_pos": self._anchor_positions.detach().cpu() if self._anchor_positions is not None else None,
                "anchor_coeffs": self._anchor_coeffs.detach().cpu(),
                "residual_coeffs": self._anchor_residual_coeffs.detach().cpu() if self._anchor_residual_coeffs is not None else None,
                "anchor_idx": self._anchor_indices.detach().cpu(),
                "anchor_w": self._anchor_weights.detach().cpu(),
                "anchor_k": self.anchor_k,
                "residual_k": self.anchor_residual_k,
                "residual_tail": self.anchor_residual_tail,
                "dct_k": self.dct_k,
                "dct_T": self.dct_T,
            }
            torch.save(anchor, os.path.join(path, "anchor_dct.pth"))

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._apply_fp16_static()
        self.active_sh_degree = self.max_sh_degree
        if self.use_dct_deform:
            if self._trajectory_coeffs is None and not self.use_anchor_dct:
                self._trajectory_coeffs = nn.Parameter(
                    torch.zeros((self.get_xyz.shape[0], self.dct_k, 3), device="cuda").requires_grad_(True)
                )
            if self.dct_use_scale and self._trajectory_coeffs_scale is None:
                self._trajectory_coeffs_scale = nn.Parameter(
                    torch.zeros((self.get_xyz.shape[0], self.dct_k, 3), device="cuda").requires_grad_(True)
                )
            if self.dct_use_rot and self._trajectory_coeffs_rot is None:
                self._trajectory_coeffs_rot = nn.Parameter(
                    torch.zeros((self.get_xyz.shape[0], self.dct_k, 4), device="cuda").requires_grad_(True)
                )
            if self.dct_use_gate and self._dct_log_alpha is None:
                self._dct_log_alpha = nn.Parameter(
                    torch.full((self.dct_k,), self.dct_gate_init, device="cuda", dtype=torch.float32).requires_grad_(True)
                )

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        
    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]
    
    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if len(group["params"]) > 1:
                continue
            if group.get("name", "") in ("dct_codebook_pos", "dct_codebook_scale", "dct_codebook_rot"):
                optimizable_tensors[group["name"]] = group["params"][0]
                continue
            if group.get("name", "") == "anchor_dct_coeffs":
                optimizable_tensors[group["name"]] = group["params"][0]
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        if self.use_dct_deform and "dct_coeffs" in optimizable_tensors:
            self._trajectory_coeffs = optimizable_tensors["dct_coeffs"]
        if self.use_dct_deform and "dct_coeffs_scale" in optimizable_tensors:
            self._trajectory_coeffs_scale = optimizable_tensors["dct_coeffs_scale"]
        if self.use_dct_deform and "dct_coeffs_rot" in optimizable_tensors:
            self._trajectory_coeffs_rot = optimizable_tensors["dct_coeffs_rot"]
        if self.use_anchor_dct and "anchor_dct_residuals" in optimizable_tensors:
            self._anchor_residual_coeffs = optimizable_tensors["anchor_dct_residuals"]
        if self.dct_use_codebook_pos and "dct_residual_pos" in optimizable_tensors:
            self._dct_codebook_residual_pos = optimizable_tensors["dct_residual_pos"]
        if self.dct_use_codebook_scale and "dct_residual_scale" in optimizable_tensors:
            self._dct_codebook_residual_scale = optimizable_tensors["dct_residual_scale"]
        if self.dct_use_codebook_rot and "dct_residual_rot" in optimizable_tensors:
            self._dct_codebook_residual_rot = optimizable_tensors["dct_residual_rot"]
        if self.dct_use_codebook_pos and self._dct_codebook_indices_pos is not None:
            self._dct_codebook_indices_pos = self._dct_codebook_indices_pos[valid_points_mask]
        if self.dct_use_codebook_scale and self._dct_codebook_indices_scale is not None:
            self._dct_codebook_indices_scale = self._dct_codebook_indices_scale[valid_points_mask]
        if self.dct_use_codebook_rot and self._dct_codebook_indices_rot is not None:
            self._dct_codebook_indices_rot = self._dct_codebook_indices_rot[valid_points_mask]
        self._deformation_accum = self._deformation_accum[valid_points_mask]
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self._deformation_table = self._deformation_table[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if len(group["params"])>1:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_deformation_table):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        # "deformation": new_deformation
       }
        if self.use_dct_deform:
            if self.dct_use_codebook_pos:
                if self._dct_codebook_residual_pos is not None:
                    d["dct_residual_pos"] = torch.zeros((new_xyz.shape[0], self.dct_k, 3), device="cuda")
            else:
                d["dct_coeffs"] = torch.zeros((new_xyz.shape[0], self.dct_k, 3), device="cuda")
            if self.use_anchor_dct and self._anchor_residual_coeffs is not None and self.anchor_residual_k > 0:
                d["anchor_dct_residuals"] = torch.zeros((new_xyz.shape[0], self.anchor_residual_k, 3), device="cuda")
            if self.dct_use_scale:
                if self.dct_use_codebook_scale:
                    if self._dct_codebook_residual_scale is not None:
                        d["dct_residual_scale"] = torch.zeros((new_xyz.shape[0], self.dct_k, 3), device="cuda")
                else:
                    d["dct_coeffs_scale"] = torch.zeros((new_xyz.shape[0], self.dct_k, 3), device="cuda")
            if self.dct_use_rot:
                if self.dct_use_codebook_rot:
                    if self._dct_codebook_residual_rot is not None:
                        d["dct_residual_rot"] = torch.zeros((new_xyz.shape[0], self.dct_k, 4), device="cuda")
                else:
                    d["dct_coeffs_rot"] = torch.zeros((new_xyz.shape[0], self.dct_k, 4), device="cuda")

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        if self.use_dct_deform and "dct_coeffs" in optimizable_tensors:
            self._trajectory_coeffs = optimizable_tensors["dct_coeffs"]
        if self.use_dct_deform and "dct_coeffs_scale" in optimizable_tensors:
            self._trajectory_coeffs_scale = optimizable_tensors["dct_coeffs_scale"]
        if self.use_dct_deform and "dct_coeffs_rot" in optimizable_tensors:
            self._trajectory_coeffs_rot = optimizable_tensors["dct_coeffs_rot"]
        if self.dct_use_codebook_pos and "dct_residual_pos" in optimizable_tensors:
            self._dct_codebook_residual_pos = optimizable_tensors["dct_residual_pos"]
        if self.dct_use_codebook_scale and "dct_residual_scale" in optimizable_tensors:
            self._dct_codebook_residual_scale = optimizable_tensors["dct_residual_scale"]
        if self.dct_use_codebook_rot and "dct_residual_rot" in optimizable_tensors:
            self._dct_codebook_residual_rot = optimizable_tensors["dct_residual_rot"]
        if self.dct_use_codebook_pos and self._dct_codebook_indices_pos is not None:
            new_idx = torch.zeros((new_xyz.shape[0],), device="cuda", dtype=self._dct_codebook_indices_pos.dtype)
            self._dct_codebook_indices_pos = torch.cat([self._dct_codebook_indices_pos, new_idx], dim=0)
        if self.dct_use_codebook_scale and self._dct_codebook_indices_scale is not None:
            new_idx = torch.zeros((new_xyz.shape[0],), device="cuda", dtype=self._dct_codebook_indices_scale.dtype)
            self._dct_codebook_indices_scale = torch.cat([self._dct_codebook_indices_scale, new_idx], dim=0)
        if self.dct_use_codebook_rot and self._dct_codebook_indices_rot is not None:
            new_idx = torch.zeros((new_xyz.shape[0],), device="cuda", dtype=self._dct_codebook_indices_rot.dtype)
            self._dct_codebook_indices_rot = torch.cat([self._dct_codebook_indices_rot, new_idx], dim=0)
        
        self._deformation_table = torch.cat([self._deformation_table,new_deformation_table],-1)
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self._deformation_accum = torch.zeros((self.get_xyz.shape[0], 3), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        if not selected_pts_mask.any():
            return
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means = torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_deformation_table = self._deformation_table[selected_pts_mask].repeat(N)
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_deformation_table)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask] 
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_deformation_table = self._deformation_table[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_deformation_table)
    
    def prune(self, max_grad, min_opacity, extent, max_screen_size):
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()

    def densify(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)
    
    def standard_constaint(self):
        
        means3D = self._xyz.detach()
        scales = self._scaling.detach()
        rotations = self._rotation.detach()
        opacity = self._opacity.detach()
        time =  torch.tensor(0).to("cuda").repeat(means3D.shape[0],1)
        means3D_deform, scales_deform, rotations_deform, _ = self._deformation(means3D, scales, rotations, opacity, time)
        position_error = (means3D_deform - means3D)**2
        rotation_error = (rotations_deform - rotations)**2 
        scaling_erorr = (scales_deform - scales)**2
        return position_error.mean() + rotation_error.mean() + scaling_erorr.mean()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    @torch.no_grad()
    def update_deformation_table(self,threshold):
        # print("origin deformation point nums:",self._deformation_table.sum())
        self._deformation_table = torch.gt(self._deformation_accum.max(dim=-1).values/100,threshold)

    def print_deformation_weight_grad(self):
        for name, weight in self._deformation.named_parameters():
            if weight.requires_grad:
                if weight.grad is None:
                    
                    print(name," :",weight.grad)
                else:
                    if weight.grad.mean() != 0:
                        print(name," :",weight.grad.mean(), weight.grad.min(), weight.grad.max())
        print("-"*50)
    
    def _plane_regulation(self):
        multi_res_grids = self._deformation.deformation_net.grid.grids
        total = 0
        # model.grids is 6 x [1, rank * F_dim, reso, reso]
        for grids in multi_res_grids:
            if len(grids) == 3:
                time_grids = []
            else:
                time_grids =  [0,1,3]
            for grid_id in time_grids:
                total += compute_plane_smoothness(grids[grid_id])
        return total

    def _time_regulation(self):
        multi_res_grids = self._deformation.deformation_net.grid.grids
        total = 0
        # model.grids is 6 x [1, rank * F_dim, reso, reso]
        for grids in multi_res_grids:
            if len(grids) == 3:
                time_grids = []
            else:
                time_grids =[2, 4, 5]
            for grid_id in time_grids:
                total += compute_plane_smoothness(grids[grid_id])
        return total
    
    def _l1_regulation(self):
                # model.grids is 6 x [1, rank * F_dim, reso, reso]
        multi_res_grids = self._deformation.deformation_net.grid.grids

        total = 0.0
        for grids in multi_res_grids:
            if len(grids) == 3:
                continue
            else:
                # These are the spatiotemporal grids
                spatiotemporal_grids = [2, 4, 5]
            for grid_id in spatiotemporal_grids:
                total += torch.abs(1 - grids[grid_id]).mean()
        return total

    def compute_regulation(self, time_smoothness_weight, l1_time_planes_weight, plane_tv_weight):
        return plane_tv_weight * self._plane_regulation() + time_smoothness_weight * self._time_regulation() + l1_time_planes_weight * self._l1_regulation()
