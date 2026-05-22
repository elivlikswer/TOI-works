import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import numpy as np

# ============================================================
#  Морфологические операции и маски
# ============================================================

MASKS = {
    "Квадрат 3x3 (8-связность)": [
        (-1, -1), (-1, 0), (-1, 1),
        ( 0, -1), ( 0, 0), ( 0, 1),
        ( 1, -1), ( 1, 0), ( 1, 1)
    ],
    "Крест 3x3 (4-связность)": [
        (0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)
    ],
    "Прямоугольник 2x2": [
        (0, 0), (0, 1), (1, 0), (1, 1)
    ]
}

def dilate(binary: np.ndarray, mask_offsets, boundary: str) -> np.ndarray:
    """Дилатация (наращивание) бинарного изображения."""
    h, w = binary.shape
    out = np.zeros_like(binary)

    if boundary == "padding":
        for y in range(h):
            for x in range(w):
                if binary[y, x] == 1:
                    for dx, dy in mask_offsets:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < w and 0 <= ny < h:
                            out[ny, nx] = 1
    else:  # ignore borders
        for y in range(h):
            for x in range(w):
                if binary[y, x] == 1:
                    fits = True
                    for dx, dy in mask_offsets:
                        nx, ny = x + dx, y + dy
                        if not (0 <= nx < w and 0 <= ny < h):
                            fits = False
                            break
                    if fits:
                        for dx, dy in mask_offsets:
                            nx, ny = x + dx, y + dy
                            out[ny, nx] = 1
    return out

def erode(binary: np.ndarray, mask_offsets, boundary: str) -> np.ndarray:
    """Эрозия (сужение) бинарного изображения."""
    h, w = binary.shape
    out = np.zeros_like(binary)

    if boundary == "padding":
        for y in range(h):
            for x in range(w):
                keep = 1
                for dx, dy in mask_offsets:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        if binary[ny, nx] == 0:
                            keep = 0
                            break
                    else:  # за границей считаем 0 -> условие не выполняется
                        keep = 0
                        break
                out[y, x] = keep
    else:  # ignore borders
        for y in range(h):
            for x in range(w):
                fits = True
                for dx, dy in mask_offsets:
                    nx, ny = x + dx, y + dy
                    if not (0 <= nx < w and 0 <= ny < h):
                        fits = False
                        break
                if fits:
                    keep = 1
                    for dx, dy in mask_offsets:
                        nx, ny = x + dx, y + dy
                        if binary[ny, nx] == 0:
                            keep = 0
                            break
                    out[y, x] = keep
                else:
                    out[y, x] = 0
    return out

def binary_to_image(binary: np.ndarray) -> Image.Image:
    """Преобразует матрицу 0/1 в ч/б PIL Image (0 – чёрный объект)."""
    # 1 -> чёрный (0), 0 -> белый (255)
    img_array = np.where(binary == 1, 0, 255).astype(np.uint8)
    return Image.fromarray(img_array, mode='L')

# ============================================================
#  Графический интерфейс
# ============================================================
class MorphApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Морфологическая обработка бинарных матриц")

        # Хранение изображений
        self.original_image = None       # PIL Image (градации серого)
        self.binary_matrix = None        # numpy 2D (0/1)
        self.display_img = None          # для PhotoImage

        # Переменные Tkinter
        self.mask_var = tk.StringVar(value=list(MASKS.keys())[0])
        self.boundary_var = tk.StringVar(value="padding")

        # Строим интерфейс
        self.create_widgets()

    def create_widgets(self):
        # Верхняя панель управления
        ctrl = ttk.Frame(self.root, padding=5)
        ctrl.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(ctrl, text="Загрузить изображение", command=self.load_image).pack(side=tk.LEFT, padx=3)
        ttk.Button(ctrl, text="Бинаризация", command=self.binarize).pack(side=tk.LEFT, padx=3)

        ttk.Separator(ctrl, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=6, fill=tk.Y)

        ttk.Label(ctrl, text="Маска:").pack(side=tk.LEFT)
        mask_menu = ttk.OptionMenu(ctrl, self.mask_var, self.mask_var.get(), *MASKS.keys())
        mask_menu.pack(side=tk.LEFT, padx=3)

        ttk.Label(ctrl, text="Границы:").pack(side=tk.LEFT, padx=(10,0))
        rb_pad = ttk.Radiobutton(ctrl, text="Расширение нулями", variable=self.boundary_var, value="padding")
        rb_ign = ttk.Radiobutton(ctrl, text="Игнорировать границы", variable=self.boundary_var, value="ignore")
        rb_pad.pack(side=tk.LEFT)
        rb_ign.pack(side=tk.LEFT, padx=3)

        ttk.Separator(ctrl, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=6, fill=tk.Y)

        ttk.Button(ctrl, text="Наращивание", command=self.apply_dilation).pack(side=tk.LEFT, padx=3)
        ttk.Button(ctrl, text="Эрозия", command=self.apply_erosion).pack(side=tk.LEFT, padx=3)
        ttk.Button(ctrl, text="Сброс", command=self.reset).pack(side=tk.LEFT, padx=3)

        # Область отображения
        self.img_frame = ttk.Frame(self.root)
        self.img_frame.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)

        self.lbl_original = ttk.Label(self.img_frame, text="Исходное изображение")
        self.lbl_original.pack(side=tk.LEFT, expand=True)

        self.lbl_processed = ttk.Label(self.img_frame, text="Обработанное изображение")
        self.lbl_processed.pack(side=tk.RIGHT, expand=True)

    def load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Изображения", "*.png *.jpg *.jpeg *.bmp *.gif"), ("Все файлы", "*.*")]
        )
        if not path:
            return
        # Загружаем и переводим в оттенки серого
        img = Image.open(path).convert('L')
        self.original_image = img
        self.binary_matrix = None
        self.show_original()

    def show_original(self):
        if self.original_image is None:
            return
        self.display_img = ImageTk.PhotoImage(self.original_image)
        self.lbl_original.configure(image=self.display_img, text="")
        self.lbl_processed.configure(image="", text="Обработанное изображение")

    def binarize(self):
        if self.original_image is None:
            return
        # Бинаризация: порог 125 ( <125 -> чёрный (1) )
        gray = np.array(self.original_image, dtype=np.uint8)
        self.binary_matrix = (gray < 125).astype(np.uint8)   # True/False -> 1/0
        self.show_binary()

    def show_binary(self):
        if self.binary_matrix is None:
            return
        # Показываем текущее бинарное изображение справа
        img = binary_to_image(self.binary_matrix)
        self.display_img = ImageTk.PhotoImage(img)
        self.lbl_processed.configure(image=self.display_img, text="")

    def apply_dilation(self):
        if self.binary_matrix is None:
            return
        mask = MASKS[self.mask_var.get()]
        boundary = self.boundary_var.get()
        self.binary_matrix = dilate(self.binary_matrix, mask, boundary)
        self.show_binary()

    def apply_erosion(self):
        if self.binary_matrix is None:
            return
        mask = MASKS[self.mask_var.get()]
        boundary = self.boundary_var.get()
        self.binary_matrix = erode(self.binary_matrix, mask, boundary)
        self.show_binary()

    def reset(self):
        # Возврат к бинарному изображению, полученному после последней бинаризации
        # Если не было бинаризации, сбрасываем к исходному серому
        if self.original_image is None:
            return
        self.binarize()
        # При желании можно сохранить исходную бинарную матрицу при бинаризации,
        # но здесь просто повторно бинаризуем оригинал.

# ============================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = MorphApp(root)
    root.mainloop()
