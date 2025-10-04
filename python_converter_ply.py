from pathlib import Path

import numpy as np
import torch
from einops import einsum, rearrange
from jaxtyping import Float
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation as R
from torch import Tensor


def construct_list_of_attributes(num_rest: int) -> list[str]:
    props = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    for i in range(num_rest):
        props.append(f"f_rest_{i}")
    props += [
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    ]
    return props


def _safe_log_scales(scales: Tensor, eps: float = 1e-8) -> Tensor:
    return torch.log(scales.clamp_min(eps))


def _safe_logit(p: Tensor, eps: float = 1e-6) -> Tensor:
    p = p.clamp(eps, 1.0 - eps)
    return torch.log(p / (1.0 - p))


def _to_wxyz_unit_quat(quat_xyzw: Tensor) -> np.ndarray:
    """
    입력: [N,4] in [x,y,z,w]
    출력: [N,4] in [w,x,y,z], 정규화 보장
    """
    # torch -> numpy
    q = quat_xyzw.detach().cpu().numpy()
    Rm = R.from_quat(q).as_matrix()
    qn = R.from_matrix(Rm).as_quat()  # [x,y,z,w]
    x, y, z, w = qn[:, 0], qn[:, 1], qn[:, 2], qn[:, 3]
    # [w,x,y,z]
    wxyz = np.stack([w, x, y, z], axis=-1)
    return wxyz.astype(np.float32)


def export_ply(
    means: Float[Tensor, "N 3"],
    scales: Float[Tensor, "N 3"],
    rotations_xyzw: Float[Tensor, "N 4"],
    harmonics: Float[Tensor, "N 3 Dsh"],  # Dsh = (deg+1)^2
    opacities: Float[Tensor, "N "],
    path: Path,
    *,
    shift_and_scale: bool = False,
    save_sh_dc_only: bool = False,  # JS 변환에 맞추려면 False로 두는게 좋음(DC+rest 모두)
    min_scale: float = 1e-8,
    min_alpha: float = 1e-6,
):
    N, C, Dsh = harmonics.shape  # C=3 (rgb)
    assert C == 3, "harmonics must be [N, 3, Dsh]"
    assert means.shape == (N, 3)
    assert scales.shape == (N, 3)
    assert rotations_xyzw.shape == (N, 4)
    assert opacities.shape == (N,)

    if shift_and_scale:
        means = means - means.median(dim=0).values
        s = means.abs().quantile(0.95, dim=0).max().clamp_min(1e-9)
        means = means / s
        scales = scales / s

    logit_opacity = _safe_logit(opacities, eps=min_alpha)  # [N]

    log_scales = _safe_log_scales(scales, eps=min_scale)  # [N,3]

    wxyz = _to_wxyz_unit_quat(rotations_xyzw)  # [N,4] float32

    f_dc = harmonics[..., 0]  # [N,3]
    f_rest = harmonics[..., 1:].reshape(N, -1)  # [N, 3*(Dsh-1)]

    num_rest = 0 if save_sh_dc_only else f_rest.shape[1]
    dtype_full = [(name, "f4") for name in construct_list_of_attributes(num_rest)]

    # === 버퍼 구성 ===
    verts = np.empty(N, dtype=dtype_full)

    zeros_normals = torch.zeros_like(means)  # nx,ny,nz -> 0
    chunks = [
        means.detach().cpu().numpy().astype(np.float32),  # x,y,z
        zeros_normals.detach().cpu().numpy().astype(np.float32),  # nx,ny,nz
        f_dc.detach().cpu().numpy().astype(np.float32),  # f_dc_0..2
    ]
    if not save_sh_dc_only:
        chunks.append(f_rest.detach().cpu().numpy().astype(np.float32))  # f_rest_*
    chunks += [
        logit_opacity.view(N, 1)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32),  # opacity (logit)
        log_scales.detach().cpu().numpy().astype(np.float32),  # scale_0..2 (log)
        wxyz.astype(np.float32),  # rot_0..3 (wxyz)
    ]

    data = np.concatenate(chunks, axis=1)
    verts[:] = list(map(tuple, data))

    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(verts, "vertex")], text=False).write(path)


# gaussian_path = "/Users/hyojungo/Downloads/_all_of_samples/dpg/Two worn pairs of leather boots lie haphazardly on a dusty barn floor, their laces tangled and their/gaussians.pt"
# # gaussian_path = "/Users/hyojungo/Downloads/_all_of_samples/T3/A snowman wearing a scarf in a winter landscape\n/gaussians.pt"
# gaussian_path = "/Users/hyojungo/Downloads/_all_of_samples/T3/A bluebird perched on a tree branch\n/gaussians.pt"
# gaussian_path = "/Users/hyojungo/Downloads/_all_of_samples/T3/A brown leather suitcase on a train platform\n/gaussians.pt"
# score: 50
def extrinsic_to_pose(R: torch.Tensor, t: torch.Tensor, dist: float = 1.0):
    """
    R: (3,3) world->camera rotation (torch.Tensor)
    t: (3,)  world->camera translation (torch.Tensor)
    dist: distance scalar to the target.
    """
    assert R.shape == (3, 3)
    assert t.shape == (3,)

    # 카메라 중심 C = -R^T t
    C = -(R.T @ t)

    # forward = R^T * [0,0,1]
    forward = R.T @ torch.tensor([0.0, 0.0, 1.0], dtype=R.dtype)

    # target = position + forward * dist
    T = C + forward * dist
    return C, T


import os

root_path = "/Users/hyojungo/Downloads/_all_of_samples/scene"
gaussian_folders = os.listdir(root_path)

for gaussian_folder in gaussian_folders:
    gaussian_path = os.path.join(root_path, gaussian_folder, "gaussians.pt")

    try:
        gaussian_data = torch.load(gaussian_path, map_location="cpu")

        export_ply(
            gaussian_data["means"],
            gaussian_data["scales"],
            gaussian_data["rotations"],
            gaussian_data["harmonics"],
            gaussian_data["opacities"],
            Path(os.path.join(root_path, gaussian_folder, "gaussians.ply")),
            save_sh_dc_only=True,
            shift_and_scale=False,
        )

        import torch

        intrinsic = gaussian_data["pred_all_intrinsic"][0][0] * 500

        fx = intrinsic[0, 0]
        fy = intrinsic[1, 1]
        cx = intrinsic[0, 2]
        cy = intrinsic[1, 2]
        print(fx, fy, cx, cy)

        # set global variable
        extrinsic = gaussian_data["pred_all_extrinsic"]
        extrinsic = extrinsic.inverse()

        Rot = extrinsic[0][6][:3, :3]
        trans = extrinsic[0][6][:3, 3]

        pos, target = extrinsic_to_pose(Rot, trans, dist=5.0)
        dist = torch.norm((gaussian_data["means"].mean(dim=0) - pos), dim=0).item()
        pos, target = extrinsic_to_pose(Rot, trans, dist=dist)

        print("position:", pos.tolist())
        print("target:", target.tolist())
        with open(os.path.join(root_path, gaussian_folder, "gaussians.json"), "w") as f:
            import json

            json.dump(
                {
                    "position": pos.tolist(),
                    "target": target.tolist(),
                    "fx": fx.item(),
                    "fy": fy.item(),
                    "cx": cx.item(),
                    "cy": cy.item(),
                },
                f,
                indent=4,
            )
    except Exception as e:
        print(e)
        pass

a = 3
