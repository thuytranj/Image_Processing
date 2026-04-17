"""
Module: Xử lý Tích chập (Convolution) - Blur & Sharpen
Lý thuyết: Tích chập là một phép toán áp dụng một kernel (ma trận nhỏ) 
lên toàn bộ ảnh để tạo ra các hiệu ứng như làm mờ hoặc làm sắc nét.
"""

import numpy as np
from scipy.ndimage import convolve
from scipy import signal


def apply_kernel(matrix, kernel, padding_mode='reflect', normalize=True):
    """
    Hàm tích chập tổng quát - áp dụng một kernel lên toàn bộ ma trận ảnh.
    
    Parameters:
    -----------
    matrix : numpy.ndarray
        Ma trận ảnh (có thể là grayscale hoặc color)
    kernel : numpy.ndarray
        Ma trận kernel để áp dụng tích chập
    padding_mode : str
        Cách xử lý biên: 'reflect', 'constant', 'nearest', 'wrap'
    normalize : bool
        Có normalize kết quả về khoảng [0, 255] hay không
    
    Returns:
    --------
    numpy.ndarray
        Ma trận sau khi áp dụng tích chập
    
    Notes:
    ------
    Tích chập (Convolution) thực hiện theo công thức:
    I'(x,y) = Σ Σ I(x+i, y+j) * K(i, j)
    
    Các bước:
    1. Duyệt qua mỗi pixel trong ảnh
    2. Áp dụng kernel lên vùng lân cận (neighborhood)
    3. Nhân từng phần tử của kernel với giá trị pixel tương ứng
    4. Cộng tất cả các tích lại
    5. Gắn kết quả cho pixel ở giữa kernel
    """
    
    # Xử lý hình ảnh color (3 kênh) - áp dụng kernel cho từng kênh
    if len(matrix.shape) == 3:
        result = np.zeros_like(matrix, dtype=np.float32)
        for channel in range(matrix.shape[2]):
            result[:, :, channel] = apply_kernel(
                matrix[:, :, channel], 
                kernel, 
                padding_mode=padding_mode,
                normalize=False  # Normalize sau cùng
            )
    else:
        # Xử lý hình ảnh grayscale hoặc từng kênh riêng
        result = convolve(matrix.astype(np.float32), kernel, mode=padding_mode)
    
    # Khi normalize=False, giữ float để bảo toàn giá trị âm/dương cho các phép như Laplacian.
    if normalize:
        result = np.clip(result, 0, 255).astype(np.uint8)
    else:
        result = result.astype(np.float32)
    
    return result


def gaussian_blur(matrix, kernel_size=5, sigma=1.0):
    """
    Làm mờ ảnh bằng Gaussian Kernel.
    
    Parameters:
    -----------
    matrix : numpy.ndarray
        Ma trận ảnh đầu vào
    kernel_size : int
        Kích thước kernel (phải là số lẻ: 3, 5, 7, 9...)
    sigma : float
        Độ lệch chuẩn (standard deviation) của Gaussian
    
    Returns:
    --------
    numpy.ndarray
        Ảnh sau khi làm mờ
    
    Notes:
    ------
    Gaussian Kernel là ma trận có giá trị tuân theo phân phối Gaussian (Bell curve).
    Công thức Gaussian 2D:
    K(x, y) = (1 / (2π*σ²)) * exp(-(x² + y²) / (2*σ²))
    
    Lợi ích: 
    - Làm mờ tự nhiên, giống với cách mắt nhìn
    - Loại bỏ noise hiệu quả
    - Giảm chi tiết, làm mềm mại ảnh
    """
    
    # Tạo Gaussian kernel
    ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    
    # Công thức Gaussian 2D
    gaussian = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    gaussian = gaussian / np.sum(gaussian)  # Normalize để tổng = 1
    
    return apply_kernel(matrix, gaussian, normalize=True)


def sharpen(matrix, kernel_type='laplacian', strength=1.0):
    """
    Làm sắc nét ảnh bằng Laplacian Kernel.
    
    Parameters:
    -----------
    matrix : numpy.ndarray
        Ma trận ảnh đầu vào
    kernel_type : str
        Loại kernel: 'laplacian' hoặc 'laplacian_diagonal'
    strength : float
        Độ mạnh của hiệu ứng làm sắc nét (0.5 - 2.0)
    
    Returns:
    --------
    numpy.ndarray
        Ảnh sau khi làm sắc nét
    
    Notes:
    ------
    Laplacian Kernel phát hiện biên cạnh bằng cách giảm hàng xóm.
    Công thức unsharp masking:
    I'(x,y) = I(x,y) + strength * Laplacian(x,y)
    
    Loại kernel:
    1. Laplacian chuẩn (4-liên kết): phìa trên, dưới, trái, phải
    2. Laplacian đường chéo (8-liên kết): thêm góc chéo
    
    Lợi ích:
    - Làm nổi bật chi tiết và biên cạnh
    - Tăng độ sắc nét ảnh
    - Làm độ tương phản cao hơn
    """
    
    # Định nghĩa các kernel Laplacian
    if kernel_type == 'laplacian':
        # Kernel Laplacian chuẩn (chỉ 4 hàng xóm)
        laplacian_kernel = np.array([
            [ 0, -1,  0],
            [-1,  4, -1],
            [ 0, -1,  0]
        ], dtype=np.float32)
    else:  # 'laplacian_diagonal'
        # Kernel Laplacian đầy đủ (8 hàng xóm)
        laplacian_kernel = np.array([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ], dtype=np.float32)
    
    # Áp dụng Laplacian để tìm biên cạnh
    laplacian_result = apply_kernel(matrix, laplacian_kernel, normalize=False)
    
    # Unsharp masking: Ảnh gốc + strength * Laplacian
    result = matrix.astype(np.float32) + strength * laplacian_result.astype(np.float32)
    
    # Normalize kết quả về [0, 255]
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result


def edge_enhance(matrix, kernel_type='sobel'):
    """
    Tăng cường biên cạnh bằng các kernel khác nhau.
    
    Parameters:
    -----------
    matrix : numpy.ndarray
        Ma trận ảnh đầu vào
    kernel_type : str
        Loại kernel: 'sobel', 'prewitt', 'roberts'
    
    Returns:
    --------
    numpy.ndarray
        Ảnh sau khi tăng cường biên cạnh
    """
    
    if kernel_type == 'sobel':
        # Sobel kernel (phát hiện gradient)
        kernel_x = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=np.float32)
        kernel_y = np.array([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=np.float32)
    
    elif kernel_type == 'prewitt':
        kernel_x = np.array([
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1]
        ], dtype=np.float32)
        kernel_y = np.array([
            [-1, -1, -1],
            [0, 0, 0],
            [1, 1, 1]
        ], dtype=np.float32)
    
    else:  # 'roberts'
        kernel_x = np.array([
            [1, 0],
            [0, -1]
        ], dtype=np.float32)
        kernel_y = np.array([
            [0, 1],
            [-1, 0]
        ], dtype=np.float32)
    
    # Áp dụng kernel theo cả hai hướng x và y
    edge_x = apply_kernel(matrix, kernel_x, normalize=False)
    edge_y = apply_kernel(matrix, kernel_y, normalize=False)
    
    # Kết hợp: Magnitude = sqrt(Gx² + Gy²)
    result = np.sqrt(edge_x.astype(np.float32)**2 + edge_y.astype(np.float32)**2)
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result
