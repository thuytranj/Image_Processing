
"""
Module xử lý I/O ảnh.
Cung cấp các hàm nền để cả nhóm sử dụng chung.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from datetime import datetime



#  PHẦN 1: ĐỌC & GHI ẢNH  (Ma trận ↔ File)
def read_image(path: str, mode: str = "color") -> np.ndarray:
    """
    Đọc ảnh từ file → trả về ma trận NumPy.
    Tham số:
        path  : đường dẫn đến file ảnh (.jpg, .png, ...)
        mode  : "color"  → BGR  (shape: H×W×3, uint8)
                "gray"   → grayscale (shape: H×W, uint8)
                "float"  → grayscale chuẩn hóa về [0,1] (float32)

    Trả về:
        np.ndarray — ma trận ảnh
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")

    if mode == "gray":
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError(f"OpenCV không đọc được file: {path}")

    if mode == "float":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    return img


def save_image(matrix: np.ndarray, path: str) -> None:
    """
    Lưu ma trận NumPy → file ảnh.
    Tự động xử lý:
      - float [0,1]  → scale lên [0,255] trước khi ghi
      - Tạo thư mục nếu chưa tồn tại
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Nếu là float, scale về uint8
    if matrix.dtype in (np.float32, np.float64):
        matrix = np.clip(matrix * 255, 0, 255).astype(np.uint8)

    success = cv2.imwrite(str(path), matrix)
    if not success:
        raise IOError(f"Không thể ghi file: {path}")

    print(f" Đã lưu ảnh: {path}")


def matrix_info(matrix: np.ndarray, label: str = "Ảnh") -> dict:
    """
    In thông tin kỹ thuật của ma trận ảnh.
    """
    info = {
        "label"  : label,
        "shape"  : matrix.shape,
        "dtype"  : str(matrix.dtype),
        "min"    : float(matrix.min()),
        "max"    : float(matrix.max()),
        "mean"   : float(matrix.mean()),
    }
    print(f"\n{'─'*40}")
    print(f"  {label}")
    print(f"  Kích thước : {info['shape']}")
    print(f"  Kiểu dữ liệu: {info['dtype']}")
    print(f"  Min / Max  : {info['min']:.2f} / {info['max']:.2f}")
    print(f"  Trung bình : {info['mean']:.2f}")
    print(f"{'─'*40}")
    return info



#  PHẦN 2: HIỂN THỊ SO SÁNH
def show_comparison(
    original    : np.ndarray,
    processed   : np.ndarray,
    title_orig  : str = "Ảnh gốc",
    title_proc  : str = "Ảnh đã xử lý",
    cmap        : str = None,
    save_path   : str = None
) -> None:
    """
    Hiển thị 2 ảnh cạnh nhau để so sánh trực quan.

    Tham số:
        original   : ma trận ảnh gốc
        processed  : ma trận ảnh sau khi xử lý
        title_orig : tiêu đề cột trái
        title_proc : tiêu đề cột phải
        cmap       : colormap ("gray" cho ảnh grayscale, None cho màu)
        save_path  : nếu truyền vào, lưu figure ra file
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"{title_orig}  vs  {title_proc}", fontsize=14, fontweight="bold")

    # OpenCV lưu ảnh màu theo thứ tự BGR → cần chuyển sang RGB để Matplotlib hiển thị đúng
    def _to_display(img):
        if img.ndim == 3 and img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    axes[0].imshow(_to_display(original), cmap=cmap)
    axes[0].set_title(title_orig, fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(_to_display(processed), cmap=cmap)
    axes[1].set_title(title_proc, fontsize=11)
    axes[1].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f" Đã lưu figure so sánh: {save_path}")

    plt.show()


def show_pipeline(images: list, titles: list, cmap=None, save_path=None) -> None:
    """
    Hiển thị toàn bộ pipeline xử lý ảnh (nhiều hơn 2 bước).

    Ví dụ: [gốc, grayscale, blur, edge] → hiện 4 ảnh trên 1 hàng.
    """
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, img, title in zip(axes, images, titles):
        if img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()



#  PHẦN 3: FORMAT REPORT
def generate_report(
    results     : list[dict],
    output_path : str = "data/output/report.md"
) -> None:
    """
    Tạo file Markdown tổng hợp kết quả toàn pipeline.

    Mỗi dict trong `results` có cấu trúc:
    {
        "operation" : "Tăng sáng",      # tên phép toán
        "author"    : "Thành viên 2",   # người thực hiện
        "params"    : {"value": 50},    # tham số đã dùng
        "before"    : np.ndarray,       # ma trận ảnh gốc
        "after"     : np.ndarray,       # ma trận ảnh sau xử lý
        "note"      : "Cộng thêm 50...", # mô tả kỹ thuật
    }
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# Báo cáo Xử lý Ảnh — Image Processing Report")
    lines.append(f"\n> Tạo tự động lúc: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("---\n")

    for i, r in enumerate(results, start=1):
        lines.append(f"## {i}. {r.get('operation', 'N/A')}")
        lines.append(f"**Thực hiện bởi:** {r.get('author', 'N/A')}\n")

        # Thông tin kỹ thuật
        if "params" in r:
            lines.append("**Tham số:**")
            for k, v in r["params"].items():
                lines.append(f"- `{k}`: {v}")
            lines.append("")

        # Thống kê ma trận trước / sau
        if "before" in r and "after" in r:
            b, a = r["before"], r["after"]
            lines.append("**Thống kê ma trận:**\n")
            lines.append("| Chỉ số | Trước xử lý | Sau xử lý |")
            lines.append("|--------|-------------|-----------|")
            lines.append(f"| Shape  | {b.shape}   | {a.shape}  |")
            lines.append(f"| Min    | {b.min():.2f} | {a.min():.2f} |")
            lines.append(f"| Max    | {b.max():.2f} | {a.max():.2f} |")
            lines.append(f"| Mean   | {b.mean():.2f} | {a.mean():.2f} |")
            lines.append("")

        if "note" in r:
            lines.append(f"**Ghi chú kỹ thuật:** {r['note']}\n")

        lines.append("---\n")

    report_text = "\n".join(lines)
    path.write_text(report_text, encoding="utf-8")
    print(f" Report đã được tạo: {path}")
