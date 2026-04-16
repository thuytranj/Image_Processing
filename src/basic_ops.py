"""
Phần này tập trung vào 3 phép biến đổi cơ bản:
1. Tăng sáng bằng phép cộng ma trận.
2. Điều chỉnh tương phản bằng phép nhân ma trận.
3. Chuyển ảnh màu sang ảnh xám bằng tổ hợp tuyến tính các kênh màu.

Mục tiêu:
- Ảnh được xem như một ma trận số.
- Mỗi pixel được xử lý trực tiếp bằng công thức toán học.
- Kết quả sau cùng phải được đưa về miền [0, 255] để phù hợp ảnh 8-bit.
"""



from __future__ import annotations
import numpy as np



def validate_image(image: np.ndarray) -> np.ndarray:
    """
    Chức năng: 
        - Kiểm tra đầu vào có đúng là một ma trận ảnh hợp lệ hay không.

    Ý tưởng: 
        - Trước khi xử lý từng pixel, cần xác nhận dữ liệu đầu vào đúng định dạng, chỉ chấp nhận:
            + Ảnh xám 2 chiều: H x W
            + Ảnh màu 3 chiều: H x W x 3

    Tham số:
        - image: Ma trận ảnh đầu vào dưới dạng numpy.ndarray.

    Kết quả:
        - Trả về biến `image` nếu dữ liệu hợp lệ.
        - Nếu không hợp lệ thì trả lỗi.
    """
    # Kiểm tra kiểu dữ liệu đầu vào.
    if not isinstance(image, np.ndarray):
        raise TypeError("image phải là một numpy.ndarray")

    # Ảnh chỉ nên có 2 chiều (grayscale) hoặc 3 chiều (ảnh màu).
    if image.ndim not in (2, 3):
        raise ValueError("image phải là ảnh xám 2D hoặc ảnh màu 3D")

    # Nếu là ảnh màu thì số kênh bắt buộc phải bằng 3.
    if image.ndim == 3 and image.shape[2] != 3:
        raise ValueError("ảnh màu phải có đúng 3 kênh")

    return image


def saturate_to_uint8(value: float) -> np.uint8:
    """
    Chức năng:
        - Đưa một giá trị số về miền [0, 255] và chuyển sang kiểu uint8.

    Ý tưởng:
        - Sau các phép cộng hoặc nhân, giá trị pixel có thể nhỏ hơn 0 hoặc lớn hơn 255.
        - Vì ảnh 8-bit chỉ lưu được trong khoảng [0, 255], ta cần chặn giá trị lại trước khi gán 
        vào ma trận kết quả.

    Tham số:
        - value: Giá trị cường độ sáng của một pixel sau khi tính toán.

    Kết quả:
        - Trả về một giá trị kiểu `np.uint8`, an toàn để ghi vào ảnh đầu ra.
    """
    if value < 0:
        return np.uint8(0)
    if value > 255:
        return np.uint8(255)
    return np.uint8(round(value))


def brighten(image: np.ndarray, value: int | float) -> np.ndarray:
    """
    Chức năng:
        - Tăng hoặc giảm độ sáng của ảnh bằng phép cộng một hằng số vào từng pixel.

    Ý tưởng:
        - Mỗi phần tử trong ma trận ảnh biểu diễn cường độ sáng của một pixel hoặc một kênh màu. 
        Khi cộng cùng một hằng số `value` cho mọi phần tử, toàn bộ ảnh sẽ sáng hơn hoặc tối hơn.
        - Công thức:
            I'(x, y) = I(x, y) + value

    Tham số:
        - image: Ảnh đầu vào, có thể là ảnh xám HxW hoặc ảnh màu BGR HxWx3.
        - value: Hằng số cộng thêm vào từng pixel.
            + `value > 0`: tăng sáng
            + `value < 0`: giảm sáng

    Kết quả:
        - Trả về một ma trận ảnh mới cùng kích thước với ảnh đầu vào, kiểu dữ liệu `uint8` và đã 
        được thay đổi độ sáng.
    """
    # Đảm bảo ảnh đầu vào hợp lệ trước khi xử lý.
    image = validate_image(image)
    height, width = image.shape[:2]

    if image.ndim == 2:
        # Tạo ma trận kết quả cùng kích thước cho ảnh xám.
        result = np.zeros((height, width), dtype=np.uint8)

        for row in range(height):
            for col in range(width):
                # Lấy giá trị pixel hiện tại rồi cộng thêm hằng số sáng.
                new_value = float(image[row, col]) + float(value)
                # Chặn kết quả về [0, 255] trước khi gán lại.
                result[row, col] = saturate_to_uint8(new_value)

        return result

    # Tạo ma trận kết quả cho ảnh màu, vẫn giữ nguyên 3 kênh BGR.
    result = np.zeros((height, width, 3), dtype=np.uint8)

    for row in range(height):
        for col in range(width):
            for channel in range(3):
                # Xử lý riêng từng kênh màu tại mỗi pixel.
                new_value = float(image[row, col, channel]) + float(value)
                result[row, col, channel] = saturate_to_uint8(new_value)

    return result


def adjust_contrast(image: np.ndarray, factor: int | float) -> np.ndarray:
    """
    Chức năng:
        - Điều chỉnh mức tương phản của ảnh bằng phép nhân mỗi pixel với một hệ số.

    Ý tưởng:
        - Nhân toàn bộ ma trận ảnh với một hệ số `factor`. Khi hệ số tăng, các giá trị cường độ 
        được kéo giãn mạnh hơn.
        - Công thức:
            I'(x, y) = I(x, y) * factor

    Tham số:
        - image: Ảnh đầu vào, có thể là ảnh xám HxW hoặc ảnh màu BGR HxWx3.
        - factor: Hệ số nhân áp dụng cho từng pixel.
            + `factor > 1`: tăng độ gắt của mức sáng
            + `0 < factor < 1`: giảm độ biến thiên mức sáng

    Kết quả:
        - Trả về một ma trận ảnh mới cùng kích thước với ảnh đầu vào, kiểu dữ liệu `uint8`, sau 
        khi đã điều chỉnh tương phản.
    """
    image = validate_image(image)

    # Hệ số âm không phù hợp với ý nghĩa điều chỉnh tương phản cơ bản trong bài này.
    if factor < 0:
        raise ValueError("factor nên là số không âm")

    height, width = image.shape[:2]

    if image.ndim == 2:
        # Tạo ma trận kết quả cho ảnh xám.
        result = np.zeros((height, width), dtype=np.uint8)

        for row in range(height):
            for col in range(width):
                # Nhân giá trị pixel với hệ số tương phản.
                new_value = float(image[row, col]) * float(factor)
                # Chặn kết quả về miền hợp lệ của ảnh 8-bit.
                result[row, col] = saturate_to_uint8(new_value)

        return result

    # Tạo ma trận kết quả cho ảnh màu BGR.
    result = np.zeros((height, width, 3), dtype=np.uint8)

    for row in range(height):
        for col in range(width):
            for channel in range(3):
                # Nhân riêng từng kênh B, G, R với cùng một hệ số.
                new_value = float(image[row, col, channel]) * float(factor)
                result[row, col, channel] = saturate_to_uint8(new_value)

    return result


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Chức năng:
        - Chuyển ảnh màu sang ảnh xám bằng công thức tổ hợp tuyến tính các kênh màu.

    Ý tưởng:
        - Ảnh màu gồm 3 kênh B, G, R. Để thu được một mức xám duy nhất cho mỗi pixel, ta kết hợp 
        3 kênh theo trọng số phản ánh độ nhạy sáng của mắt người.
        - Công thức:
            Gray = 0.114 * B + 0.587 * G + 0.299 * R

    Tham số:
        - image: Ảnh đầu vào. 
            + Nếu là ảnh màu HxWx3 thì hàm sẽ chuyển sang ảnh xám.
            + Nếu đã là ảnh xám HxW thì hàm trả về một bản sao của ảnh đó.

    Kết quả:
        - Trả về ảnh xám 2 chiều HxW, kiểu dữ liệu `uint8`.
    """
    image = validate_image(image)

    # Nếu đầu vào đã là ảnh xám, trả về bản sao để không làm thay đổi dữ liệu gốc.
    if image.ndim == 2:
        return image.copy()

    height, width = image.shape[:2]
    # Ảnh xám đầu ra chỉ còn 1 giá trị sáng cho mỗi pixel.
    result = np.zeros((height, width), dtype=np.uint8)

    for row in range(height):
        for col in range(width):
            # Ảnh đầu vào dùng thứ tự BGR theo cách đọc mặc định của OpenCV.
            blue = float(image[row, col, 0])
            green = float(image[row, col, 1])
            red = float(image[row, col, 2])

            # Tính mức xám theo công thức viết theo thứ tự BGR.
            gray_value = 0.114 * blue + 0.587 * green + 0.299 * red
            result[row, col] = saturate_to_uint8(gray_value)

    return result


__all__ = [
    "brighten",
    "adjust_contrast",
    "to_grayscale",
]
