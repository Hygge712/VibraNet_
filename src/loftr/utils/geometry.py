import torch


@torch.no_grad()
def warp_kpts(kpts0, depth0, depth1, T_0to1, K0, K1):
    """ Warp kpts0 from I0 to I1 with depth, K and Rt
    Also check covisibility and depth consistency.
    Depth is consistent if relative error < 0.2 (hard-coded).
    对关键点的深度信息采样、投影、重投影和双目视图间的一致性检查。
    Args:
        kpts0 (torch.Tensor): [N, L, 2] - <x, y>,
        depth0 (torch.Tensor): [N, H, W],
        depth1 (torch.Tensor): [N, H, W],
        T_0to1 (torch.Tensor): [N, 3, 4], 从I0到I1的刚体变换矩阵
        K0 (torch.Tensor): [N, 3, 3],
        K1 (torch.Tensor): [N, 3, 3],
    Returns:
        calculable_mask (torch.Tensor): [N, L]
        warped_keypoints0 (torch.Tensor): [N, L, 2] <x0_hat, y1_hat>
    """
    kpts0_long = kpts0.round().long()  # 关键点的深度采样与有效性检查

    # Sample depth, get calculable_mask on depth != 0  # 关键点深度采样
    kpts0_depth = torch.stack(
        [depth0[i, kpts0_long[i, :, 1], kpts0_long[i, :, 0]] for i in range(kpts0.shape[0])], dim=0
    )  # (N, L)
    nonzero_mask = kpts0_depth != 0

    # Unproject  # 关键点逆投影到相机坐标系，在*depth后得到相机坐标系下的3D坐标。
    kpts0_h = torch.cat([kpts0, torch.ones_like(kpts0[:, :, [0]])], dim=-1) * kpts0_depth[..., None]  # (N, L, 3)
    kpts0_cam = K0.inverse() @ kpts0_h.transpose(2, 1)  # (N, 3, L)

    # Rigid Transform  # 通过刚体变换到世界坐标系
    w_kpts0_cam = T_0to1[:, :3, :3] @ kpts0_cam + T_0to1[:, :3, [3]]  # (N, 3, L)  # ==>旋转矩阵@世界坐标系 + 平移变量
    w_kpts0_depth_computed = w_kpts0_cam[:, 2, :]  # 关键点在世界坐标系中Z轴的深度值

    # Project  # 在I1上重投影
    w_kpts0_h = (K1 @ w_kpts0_cam).transpose(2, 1)  # (N, L, 3)
    w_kpts0 = w_kpts0_h[:, :, :2] / (w_kpts0_h[:, :, [2]] + 1e-4)  # (N, L, 2), +1e-4 to avoid zero depth

    # Covisible Check： 筛选投影后落在图像范围内的关键点
    h, w = depth1.shape[1:3]
    covisible_mask = (w_kpts0[:, :, 0] > 0) * (w_kpts0[:, :, 0] < w - 1) * (w_kpts0[:, :, 1] > 0) * (
                w_kpts0[:, :, 1] < h - 1)

    w_kpts0_long = w_kpts0.long()  # 在I1的深度图上采样重投影的关键点深度
    w_kpts0_long[~covisible_mask, :] = 0  # 对于不可视的点的对应索引设置为0，避免非法内存访问

    w_kpts0_depth = torch.stack(
        [depth1[i, w_kpts0_long[i, :, 1], w_kpts0_long[i, :, 0]] for i in range(w_kpts0_long.shape[0])], dim=0
    )  # (N, L)
    consistent_mask = ((w_kpts0_depth - w_kpts0_depth_computed) / w_kpts0_depth).abs() < 0.2  # 一致性检查
    valid_mask = nonzero_mask * covisible_mask * consistent_mask  # 有效性掩码结果

    return valid_mask, w_kpts0
