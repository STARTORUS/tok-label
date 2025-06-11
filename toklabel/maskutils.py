from typing import List, Tuple, Dict, Optional, Union
from label_studio_sdk.converter.brush import decode_rle, encode_rle
import numpy as np
import cv2
import os
from sunist2.script.camera import K,R,distCoeffs,T,project_surface, rotate_around_z_axis, get_visible_points, project
import matplotlib.pyplot as plt 

CAMERA_ENV = {'image_param': {'h_camera':800, 'w_camera':1280},
              'position_param': {'origin_camera': np.array([-1.22091814, -0.14415186,  0.00157039]), 'center_pillar_radius': 0.209 },
              'project_param': {'K':K, 'distCoeffs':distCoeffs, 'R':R, 'T':T}
              }

def rle_to_mask(rle: str|List[str], shape):
    """
    将 RLE 编码转为二值 mask。
    
    参数:
    ----
    rle: str or list of int
        RLE 编码，形如 "4 3 10 2" 或 [4, 3, 10, 2]
    shape: (height, width)
        图像的尺寸，用于还原掩码的二维形状。
    
    返回:
    ----
    np.ndarray: 与 shape 相同的二值掩码
    """
    if isinstance(rle, str):
        rle = list(map(int, rle.strip().split()))
    mask_1d = decode_rle(rle, False)
    height, width = shape
    mask = np.reshape(mask_1d, [height, width, 4])[:, :, 3]
    return mask

def mask_to_rle(mask: np.ndarray, normalize: bool = False):
    """
    将mask编码为Label Studio的rle格式

    参数:
    ----
    mask: ny.ndarray
        二维 numpy 数组格式的二值掩码，且必须为 uint8 or int 类型
    normalize: bool
        mask应为0和255组成的灰度图。如果为 True，则将 mask 映射到 [0, 255] 范围
    
    返回:
    ----
    np.ndarray: 与 shape 相同的二值掩码

    """
    # 将mask转化为处理为255
    mask = np.array((np.array(mask) > 128) * 255, dtype=np.uint8)
    array = mask.ravel()
    array = np.repeat(array, 4)
    rle = encode_rle(array)
    return rle

def pixel_to_percent(points: Union[np.ndarray, List, Tuple], shape: Tuple[int, int] = (800, 1280)):
    """
    将像素坐标转换为百分比坐标。支持单点(keypoint)、点列(polygon)。
    """
    h, w = shape

    arr = np.array(points, dtype=np.float32)

    if arr.ndim == 1 and arr.shape[0] == 2:  # 单个点
        arr = arr[None, :]  # 变为 (1, 2)

    if arr.shape[-1] != 2:
        raise ValueError("输入数据必须为形如 (N, 2) 或 (2,) 的点集")

    percent = arr.copy()
    percent[:, 0] = np.round((arr[:, 0] / w) * 100, 2)
    percent[:, 1] = np.round((arr[:, 1] / h) * 100, 2)

    return percent.tolist() if len(percent) > 1 else percent[0].tolist()

def percent_to_pixel(points: Union[np.ndarray, List, Tuple], shape: Tuple[int, int] = (800, 1280)):
    """
    将百分比坐标还原为像素坐标。支持单点(keypoint)、点列(polygon)。
    """
    h, w = shape

    arr = np.array(points, dtype=np.float32)

    if arr.ndim == 1 and arr.shape[0] == 2:
        arr = arr[None, :]

    if arr.shape[-1] != 2:
        raise ValueError("输入数据必须为形如 (N, 2) 或 (2,) 的点集")

    pixel = arr.copy()
    pixel[:, 0] = np.round((arr[:, 0] / 100) * w)
    pixel[:, 1] = np.round((arr[:, 1] / 100) * h)

    return pixel.astype(int).tolist() if len(pixel) > 1 else pixel[0].astype(int).tolist()

def overlay_mask_on_image(image:np.ndarray, mask:np.ndarray, color=(0, 0, 255), alpha=0.5):
    """
    将 mask 叠加到 BGR 图像上，用于可视化。
    
    参数:
    ----
    image: np.ndarray
        原始图像，BGR格式，shape=(H, W, 3)
    mask: np.ndarray
        二值掩码，shape=(H, W)
    color: tuple
        mask 区域的颜色，BGR格式，默认为(0, 0, 255)
    alpha: float
        透明度，0为完全透明，1为完全不透明，默认为0.5
    
    返回:
    ----
    overlaid_image: np.ndarray
        叠加了 mask 的图像
    """
    overlay = image.copy()
    mask_binary = (mask > 0).astype(np.uint8)
    
    # 创建颜色图层
    colored_mask = np.zeros_like(image, dtype=np.uint8)
    colored_mask[mask_binary == 1] = color

    # 叠加
    cv2.addWeighted(colored_mask, alpha, overlay, 1 - alpha, 0, dst=overlay)

    return overlay

def save_mask(mask: np.ndarray, path: str, normalize: bool = False):
    """
    将二值 mask 保存为图像文件（PNG/JPG）
    
    参数:
    ----
    mask : np.ndarray
        2D numpy 数组，值为 0 或 1（或 0~255），表示掩码
    path : str
        要保存的完整路径，包含文件名及扩展名，如 "mask.png" 或 "mask.jpg"
    normalize : bool
        如果为 True，则将 mask 映射到 [0, 255] 范围
    
    返回:
    ----
    None
    """
    if normalize:
        mask = (mask * 255).astype(np.uint16)
    else:
        mask = mask.astype(np.uint16)

    mask = np.clip(mask, 0, 255).astype(np.uint8)
    # 确保目录存在
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # 保存图像
    cv2.imwrite(path, mask)

def load_mask(path, threshold:np.uint8 =None):
    """
    从图像文件加载掩码，转为 numpy 数组

    参数:
    ----
    path : str
        掩码图像的文件路径(.png / .jpg )
    threshold : np.uint8
        默认为None，保持mask原数值。否则，根据阈值转化为0和1组成的二值 numpy 数组
    返回:
    ----
    mask : np.ndarray
        掩码图像 (uint8)
    """
    # 读取图像为灰度图
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load image at path: {path}")

    # 应用阈值转为二值 mask
    if threshold:
        mask = np.array((np.array(mask) > threshold) * 255, dtype=np.uint8)
    else:
        mask = img.astype(np.uint8)
    
    return mask

def build_lcfs(center: Tuple[float, float],
               a: float,
               k: float,
               delta_u: float,
               delta_l: float,
               theta_samples: int = 400) -> np.ndarray:
    """
    生成 D 形等离子体极限轮廓 (last-closed flux surface).

    返回 shape=(N, 2) 的 ndarray, 列顺序 (R, Z)
    """
    R0, Z0 = center
    θ = np.linspace(-np.pi, np.pi, theta_samples, endpoint=False)

    # 分段三角度 δ(θ)：上半圈用 δ_u，下半圈用 δ_l
    δ = np.where(np.sin(θ) >= 0, delta_u, delta_l)

    R = R0 + a * np.cos(θ + δ * np.sin(θ))
    Z = Z0 + k * a * np.sin(θ)
    return np.vstack([R, Z]).T          # (N,2)

def generate_plasma_center(center:Tuple[float, float],
                         shape:Optional[Tuple] = None,
                         angle:Optional[Tuple] = (0.5969, 2.7960),
                         env: Dict[str,any] = CAMERA_ENV):
    '''
        根据指定参数生成等离子体截面mask

        参数：
        ----
        center:Tuple
            截面质心在真空室坐标中的位置 (m)
        shape: (height, width)
            图像的尺寸，用于还原掩码的二维形状。
        angle: (phi_left, phi_right)
            左右两个mask截面的投影角度
        返回：
        ----
        mask : tuple(np.ndarray)
            左右两张掩码图像的元组 (uint8)
    '''
    if shape is None:
        H, W = env["image_param"]["h_camera"], env["image_param"]["w_camera"]
    else:
        H, W = shape

    surface = np.array(center)
    surface = np.expand_dims(np.array(center), axis=0)
    pts3d_sec = np.c_[np.zeros((surface.shape[0],1)), surface]

    project_param = env["project_param"]
    position_param = env["position_param"]
    centers = []
    for phi in angle:
        pts2d = project(
                  get_visible_points(
                     rotate_around_z_axis(pts3d_sec, phi),
                     **position_param),
                  **project_param)
        pts2d = pts2d.reshape(2)
        centers.append(pts2d)
    return tuple(centers)

def generate_plasma_mask(center:Tuple[float, float],
                         a: float,
                         k: float,
                         delta_u: float,
                         delta_l: float,
                         theta_samples: Optional[int] = 400,
                         shape:Optional[Tuple] = None,
                         angle:Optional[Tuple] = (0.5969, 2.7960),
                         env: Dict[str,any] = CAMERA_ENV):
    '''
        根据指定参数生成等离子体截面mask

        参数：
        ----
        center:Tuple
            截面质心在真空室坐标中的位置 (m)
        a: float
            等离子体小半径 (m)
        k: float
            伸长比
        delta_u: float
            上三角度
        delta_l: float
            下三角度
        theta_samples: int
            采样点数
        shape: (height, width)
            图像的尺寸，用于还原掩码的二维形状。
        angle: (phi_left, phi_right)
            左右两个mask截面的投影角度
        返回：
        ----
        mask : tuple(np.ndarray)
            左右两张掩码图像的元组 (uint8)
    '''
    if shape is None:
        H, W = env["image_param"]["h_camera"], env["image_param"]["w_camera"]
    else:
        H, W = shape

    surface = build_lcfs(center=center,
                        a=a, k=k,
                        delta_u=delta_u, delta_l=delta_l,
                        theta_samples=theta_samples)
    pts3d_sec = np.c_[np.zeros((surface.shape[0],1)), surface]

    project_param = env["project_param"]
    position_param = env["position_param"]
    masks = []
    for phi in angle:
        pts2d = project(
                  get_visible_points(
                     rotate_around_z_axis(pts3d_sec, phi),
                     **position_param),
                  **project_param)
        pts2d = pts2d.astype(np.int32).reshape(-1,1,2)
        m = np.zeros((H, W), np.uint8)
        cv2.fillPoly(m, [pts2d], 255)
        masks.append(m)
    return tuple(masks)

def generate_plasma_polygon(
        center: Tuple[float, float],
        a: float,
        k: float,
        delta_u: float,
        delta_l: float,
        theta_samples: int = 400,
        shape: Optional[Tuple[int, int]] = None,           # (H, W)
        angle: Tuple[float, float] =(0.5969, 2.7960),
        env: Dict[str, any] = CAMERA_ENV
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    返回两个截面（φ_left, φ_right）的像素多边形顶点列表
    """

    if shape is None:
        H, W = env["image_param"]["h_camera"], env["image_param"]["w_camera"]
    else:
        H, W = shape

    surface = build_lcfs(
        center=center, a=a, k=k,
        delta_u=delta_u, delta_l=delta_l,
        theta_samples=theta_samples
    )
    pts3d_sec = np.c_[np.zeros((surface.shape[0], 1)), surface]  # (x=0, y=R, z=Z)

    project_param   = env["project_param"]
    position_param  = env["position_param"]

    polygons: List[np.ndarray] = []
    for phi in angle:
        # → 旋转 → 可见性过滤 → 投影到 2‑D
        pts2d = project(
            get_visible_points(
                rotate_around_z_axis(pts3d_sec, phi),
                **position_param,
            ),
            **project_param,
        ).astype(np.int32)

        
        hull = cv2.convexHull(pts2d).reshape(-1, 2)  # (N, 2)
        polygons.append(hull)

    left_poly, right_poly = polygons
    return left_poly, right_poly


def overlay_polygon_on_image(
    image: np.ndarray,
    polygon: Union[np.ndarray, List[Tuple[int, int]]],
    color: Tuple[int, int, int] = (0, 255, 0),  # BGR
    thickness: int = 2,
    alpha: float = 0.4,
    fill: bool = False,
) -> np.ndarray:
    """Overlay a polygon on a BGR image. Supports ndarray, list."""
    overlay = image.copy()
    H, W = image.shape[:2]

    pts = np.array(polygon, dtype=np.int32).reshape(-1, 1, 2)

    if fill:
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, [pts], color=color)
        cv2.addWeighted(mask, alpha, overlay, 1 - alpha, 0, dst=overlay)
    else:
        cv2.polylines(overlay, [pts], isClosed=True, color=color, thickness=thickness)

    return overlay

