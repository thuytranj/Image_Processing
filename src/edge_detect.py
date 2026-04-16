"""
Edge Detection Module - Phát hiện biên cạnh.

Sử dụng toán tử Sobel để tìm biên cạnh trong ảnh.
Ý tưởng: Tính gradient theo hướng X và Y, sau đó kết hợp để tìm ranh giới cường độ sáng.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple


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
    # Bước 1: Sử dụng Sobel để tính gradient
    Gx, Gy, magnitude = sobel_edges(image)
    
    # Bước 2: Ngưỡng hóa đơn giản (chỉ lấy gradient mạnh)
    edges = np.zeros_like(magnitude)
    edges[magnitude > high_threshold] = 255
    
    return edges
