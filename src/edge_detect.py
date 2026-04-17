"""
Edge Detection Module - Phát hiện biên cạnh.

Sử dụng toán tử Sobel để tìm biên cạnh trong ảnh.
Ý tưởng: Tính gradient theo hướng X và Y, sau đó kết hợp để tìm ranh giới cường độ sáng.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple


def _to_grayscale_float(image: np.ndarray) -> np.ndarray:
    """Convert input image to grayscale float32."""
    if image.ndim == 3:
        return (0.114 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.299 * image[:, :, 2]).astype(np.float32)
    return image.astype(np.float32)


def _gaussian_kernel(size: int = 5, sigma: float = 1.4) -> np.ndarray:
    """Create a normalized 2D Gaussian kernel."""
    if size % 2 == 0:
        raise ValueError("size phải là số lẻ")
    half = size // 2
    x, y = np.mgrid[-half:half + 1, -half:half + 1]
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel = kernel / np.sum(kernel)
    return kernel.astype(np.float32)


def _non_maximum_suppression(magnitude: np.ndarray, angle_deg: np.ndarray) -> np.ndarray:
    """Thin edges by keeping only local maxima along gradient direction."""
    h, w = magnitude.shape
    out = np.zeros((h, w), dtype=np.float32)

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            angle = angle_deg[i, j]

            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                q = magnitude[i, j + 1]
                r = magnitude[i, j - 1]
            elif 22.5 <= angle < 67.5:
                q = magnitude[i + 1, j - 1]
                r = magnitude[i - 1, j + 1]
            elif 67.5 <= angle < 112.5:
                q = magnitude[i + 1, j]
                r = magnitude[i - 1, j]
            else:
                q = magnitude[i - 1, j - 1]
                r = magnitude[i + 1, j + 1]

            if magnitude[i, j] >= q and magnitude[i, j] >= r:
                out[i, j] = magnitude[i, j]

    return out


def _double_threshold(image: np.ndarray, low_threshold: float, high_threshold: float) -> Tuple[np.ndarray, int, int]:
    """Classify pixels into strong, weak and non-edge."""
    strong = 255
    weak = 75
    out = np.zeros_like(image, dtype=np.uint8)

    strong_i, strong_j = np.where(image >= high_threshold)
    weak_i, weak_j = np.where((image >= low_threshold) & (image < high_threshold))

    out[strong_i, strong_j] = strong
    out[weak_i, weak_j] = weak
    return out, weak, strong


def _hysteresis(image: np.ndarray, weak: int, strong: int) -> np.ndarray:
    """Promote weak edges connected to strong edges, suppress others."""
    h, w = image.shape
    out = image.copy()

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if out[i, j] == weak:
                neighborhood = out[i - 1:i + 2, j - 1:j + 2]
                if np.any(neighborhood == strong):
                    out[i, j] = strong
                else:
                    out[i, j] = 0

    return out


# ─────── PHẦN 1: HÀM TÍCH CHẬP CHUNG ───────
def apply_kernel(image: np.ndarray, kernel: np.ndarray, padding: str = "same") -> np.ndarray:
    """
    Hàm tích chập (Convolution) chung.
    Áp dụng một kernel (ma trận nhỏ) lên toàn bộ ảnh để tạo hiệu ứng xử lý.

    Thuyết minh:
        - Tích chập là phép toán cơ bản trong xử lý ảnh.
        - Tại mỗi điểm (i, j) trong ảnh, ta lấy vùng 3×3 (hoặc kích thước kernel)
          xung quanh nó, nhân từng phần tử với kernel tương ứng, rồi cộng lại.
        - Kết quả là một giá trị mới đại diện cho "đặc trưng" tại khu vực đó.

    Công thức:
        output[i, j] = Σ Σ image[i+di, j+dj] * kernel[di, dj]
        (tính tổng trên toàn bộ kernel)

    Tham số:
        image  : ảnh đầu vào (HxW hoặc HxWx3 cho ảnh màu)
        kernel : ma trận kernel (thường là 3×3 hoặc 5×5)
        padding: "same" → giữ kích thước ảnh như cũ (padding 0)
                "valid" → kích thước nhỏ hơn (không dùng padding)

    Kết quả:
        float32 array - ảnh sau tích chập (có thể âm hoặc > 255)
    """
    if image.ndim == 3:
        # Nếu là ảnh màu, chuyển sang xám trước
        # (theo công thức BGR → Gray như trong basic_ops.py)
        h, w = image.shape[:2]
        gray = np.zeros((h, w), dtype=np.float32)
        gray = 0.114 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.299 * image[:, :, 2]
        image = gray.astype(np.float32)
    else:
        image = image.astype(np.float32)

    h, w = image.shape
    kh, kw = kernel.shape
    kernel = kernel.astype(np.float32)

    # Tính offset (vị trí phần tử giữa của kernel)
    offset_h = kh // 2
    offset_w = kw // 2

    if padding == "same":
        # Padding với 0 để giữ kích thước
        padded = np.zeros((h + 2 * offset_h, w + 2 * offset_w), dtype=np.float32)
        padded[offset_h:offset_h + h, offset_w:offset_w + w] = image
        image = padded
        h, w = image.shape

    output = np.zeros((h - kh + 1, w - kw + 1), dtype=np.float32)

    # Tích chập từng vị trí
    for i in range(h - kh + 1):
        for j in range(w - kw + 1):
            # Lấy vùng 3×3 (hoặc kích thước kernel)
            region = image[i:i + kh, j:j + kw]
            # Nhân từng phần tử với kernel rồi cộng lại
            output[i, j] = np.sum(region * kernel)

    return output


# ─────── PHẦN 2: PHÁT HIỆN BIÊN CẠNH BẰNG SOBEL ───────
def sobel_edges(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Phát hiện biên cạnh sử dụng toán tử Sobel.

    Thuyết minh:
        - Sobel là một toán tử phát hiện biên cạnh phổ biến.
        - Ý tưởng: tính gradient (độ thay đổi) của cường độ sáng trong 2 hướng X và Y.
        - Nơi gradient lớn là nơi có sự thay đổi mạnh → biên cạnh.

    Kernels:
        Gx = [[-1, 0, 1],       Gy = [[-1, -2, -1],
              [-2, 0, 2],               [0,  0,  0],
              [-1, 0, 1]]               [1,  2,  1]]

    Công thức:
        Gx = Sobel_X * image  (gradient theo hướng ngang)
        Gy = Sobel_Y * image  (gradient theo hướng dọc)
        G  = sqrt(Gx² + Gy²)  (magnitude - cường độ gradient)

    Tham số:
        image : ảnh đầu vào (xám HxW hoặc màu HxWx3)

    Kết quả:
        Tuple[Gx, Gy, G] - gradient X, gradient Y, magnitude (đều là uint8 [0, 255])
    """
    # Các kernel Sobel theo công thức chuẩn
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32)

    sobel_y = np.array([[-1, -2, -1],
                        [0,  0,  0],
                        [1,  2,  1]], dtype=np.float32)

    # Tính gradient theo X và Y bằng tích chập
    Gx = apply_kernel(image, sobel_x, padding="same")
    Gy = apply_kernel(image, sobel_y, padding="same")

    # Tính magnitude (cường độ gradient)
    magnitude = np.sqrt(Gx**2 + Gy**2)

    # Chuẩn hóa về [0, 255] để có thể lưu thành ảnh
    magnitude_normalized = (magnitude / magnitude.max() * 255).astype(np.uint8) if magnitude.max() > 0 else magnitude.astype(np.uint8)
    
    # Chuẩn hóa Gx, Gy để hiển thị (chuyển từ âm/dương sang [0, 255])
    Gx_normalized = ((Gx - Gx.min()) / (Gx.max() - Gx.min()) * 255).astype(np.uint8) if Gx.max() > Gx.min() else Gx.astype(np.uint8)
    Gy_normalized = ((Gy - Gy.min()) / (Gy.max() - Gy.min()) * 255).astype(np.uint8) if Gy.max() > Gy.min() else Gy.astype(np.uint8)

    return Gx_normalized, Gy_normalized, magnitude_normalized


def canny_edges(image: np.ndarray, low_threshold: float = 50, high_threshold: float = 150) -> np.ndarray:
    """
    Phát hiện biên cạnh sử dụng thuật toán Canny (nâng cao).
    
    Note: Phiên bản đơn giản - không bao gồm đầy đủ non-maximum suppression và hysteresis.
    
    Tham số:
        image: ảnh đầu vào
        low_threshold: ngưỡng thấp
        high_threshold: ngưỡng cao
    
    Kết quả:
        ảnh biên cạnh nhị phân (uint8)
    """
    # Bước 1: Gaussian blur để giảm nhiễu
    gray = _to_grayscale_float(image)
    blur_kernel = _gaussian_kernel(size=5, sigma=1.4)
    smoothed = apply_kernel(gray, blur_kernel, padding="same")

    # Bước 2: Tính gradient bằng Sobel
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1],
                        [0,  0,  0],
                        [1,  2,  1]], dtype=np.float32)

    gx = apply_kernel(smoothed, sobel_x, padding="same")
    gy = apply_kernel(smoothed, sobel_y, padding="same")

    magnitude = np.hypot(gx, gy)
    if magnitude.max() > 0:
        magnitude = (magnitude / magnitude.max()) * 255.0

    angle = np.rad2deg(np.arctan2(gy, gx))
    angle[angle < 0] += 180

    # Bước 3: Non-maximum suppression
    nms = _non_maximum_suppression(magnitude, angle)

    # Bước 4: Double threshold
    thresholded, weak, strong = _double_threshold(nms, low_threshold, high_threshold)

    # Bước 5: Edge tracking by hysteresis
    edges = _hysteresis(thresholded, weak, strong)
    return edges.astype(np.uint8)
