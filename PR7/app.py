import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os


class BinarizationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ЛР7: Пороговая бинаризация")
        self.root.geometry("1280x820")
        self.root.resizable(True, True)

        self.original_image = None
        self.gray_image = None
        self.binary_image = None
        self.histogram = None
        self.current_threshold = 128
        self.last_method = "-"

        self.photo_refs = {
            "original": None,
            "gray": None,
            "binary": None,
        }

        self.stats_text = tk.StringVar(value="Статистика появится после обработки изображения.")
        self.manual_result_text = tk.StringVar(value="Ручной порог: -")
        self.otsu_result_text = tk.StringVar(value="Оптимальный порог (Оцу): -")

        self.create_widgets()

    def create_widgets(self):
        control_frame = tk.Frame(self.root, padx=8, pady=8)
        control_frame.pack(fill=tk.X)

        self.load_btn = tk.Button(
            control_frame,
            text="Открыть изображение",
            command=self.load_image,
            width=18,
            bg="#1976D2",
            fg="white",
        )
        self.load_btn.pack(side=tk.LEFT, padx=5)

        self.file_label = tk.Label(control_frame, text="Файл не выбран", fg="gray")
        self.file_label.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        self.save_btn = tk.Button(
            control_frame,
            text="Сохранить бинарное",
            command=self.save_image,
            state=tk.DISABLED,
            bg="#EF6C00",
            fg="white",
        )
        self.save_btn.pack(side=tk.RIGHT, padx=5)

        self.notebook = ttk.Notebook(self.root)
        self.notebook.configure(height=240)
        self.notebook.pack(fill=tk.X, padx=10, pady=(0, 6))

        controls_tab = ttk.Frame(self.notebook)
        stats_tab = ttk.Frame(self.notebook)
        calc_tab = ttk.Frame(self.notebook)
        self.notebook.add(controls_tab, text="Управление")
        self.notebook.add(stats_tab, text="Статистика")
        self.notebook.add(calc_tab, text="Вычисления")

        manual_frame = tk.LabelFrame(controls_tab, text="Ручная бинаризация", padx=8, pady=8)
        manual_frame.pack(fill=tk.X, padx=8, pady=8)

        threshold_frame = tk.Frame(manual_frame)
        threshold_frame.pack(fill=tk.X)

        tk.Label(threshold_frame, text="Порог t (0-255):").pack(side=tk.LEFT, padx=(0, 5))

        self.threshold_var = tk.IntVar(value=128)
        self.threshold_slider = ttk.Scale(
            threshold_frame,
            from_=0,
            to=255,
            orient=tk.HORIZONTAL,
            variable=self.threshold_var,
            command=self.on_threshold_change,
        )
        self.threshold_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        self.threshold_spinbox = tk.Spinbox(
            threshold_frame,
            from_=0,
            to=255,
            textvariable=self.threshold_var,
            width=6,
            command=self.on_threshold_spin,
        )
        self.threshold_spinbox.pack(side=tk.LEFT, padx=5)

        self.apply_manual_btn = tk.Button(
            threshold_frame,
            text="Применить вручную",
            command=self.apply_manual_threshold,
            width=18,
        )
        self.apply_manual_btn.pack(side=tk.LEFT, padx=5)

        tk.Label(manual_frame, textvariable=self.manual_result_text, anchor="w").pack(fill=tk.X, pady=(6, 0))

        auto_frame = tk.LabelFrame(controls_tab, text="Автоматическая бинаризация", padx=8, pady=8)
        auto_frame.pack(fill=tk.X, padx=8, pady=(0, 8))

        self.otsu_btn = tk.Button(
            auto_frame,
            text="Применить метод Оцу",
            command=self.apply_otsu,
            bg="#2E7D32",
            fg="white",
            width=24,
        )
        self.otsu_btn.pack(anchor="w")

        tk.Label(auto_frame, textvariable=self.otsu_result_text, anchor="w").pack(fill=tk.X, pady=(8, 0))

        stats_title = tk.Label(stats_tab, text="Текущие метрики", font=("Arial", 10, "bold"), anchor="w")
        stats_title.pack(fill=tk.X, padx=10, pady=(10, 4))

        self.stats_label = tk.Label(
            stats_tab,
            textvariable=self.stats_text,
            justify=tk.LEFT,
            anchor="nw",
            font=("Consolas", 10),
            bg="#F7F7F7",
            relief=tk.GROOVE,
            padx=10,
            pady=10,
        )
        self.stats_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        calc_title = tk.Label(calc_tab, text="Подробные вычисления метода Оцу", font=("Arial", 10, "bold"), anchor="w")
        calc_title.pack(fill=tk.X, padx=10, pady=(10, 4))

        calc_container = tk.Frame(calc_tab)
        calc_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        self.calc_text = tk.Text(
            calc_container,
            wrap=tk.NONE,
            font=("Consolas", 9),
            bg="#F7F7F7",
            relief=tk.GROOVE,
            padx=8,
            pady=8,
        )
        self.calc_text.grid(row=0, column=0, sticky="nsew")

        calc_scroll_y = ttk.Scrollbar(calc_container, orient=tk.VERTICAL, command=self.calc_text.yview)
        calc_scroll_y.grid(row=0, column=1, sticky="ns")

        calc_scroll_x = ttk.Scrollbar(calc_container, orient=tk.HORIZONTAL, command=self.calc_text.xview)
        calc_scroll_x.grid(row=1, column=0, sticky="ew")

        calc_container.grid_rowconfigure(0, weight=1)
        calc_container.grid_columnconfigure(0, weight=1)

        self.calc_text.configure(yscrollcommand=calc_scroll_y.set, xscrollcommand=calc_scroll_x.set)
        self.set_calculation_report("Отчет появится после запуска метода Оцу.")

        image_frame = tk.Frame(self.root)
        image_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 6))

        image_frame.grid_columnconfigure(0, weight=1)
        image_frame.grid_columnconfigure(1, weight=1)
        image_frame.grid_columnconfigure(2, weight=1)
        image_frame.grid_rowconfigure(0, weight=1)

        original_frame = tk.LabelFrame(image_frame, text="Исходное (RGB)")
        original_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 4), pady=4)
        self.original_canvas = tk.Canvas(original_frame, bg="#ECEFF1")
        self.original_canvas.pack(fill=tk.BOTH, expand=True)

        gray_frame = tk.LabelFrame(image_frame, text="Полутоновое")
        gray_frame.grid(row=0, column=1, sticky="nsew", padx=4, pady=4)
        self.gray_canvas = tk.Canvas(gray_frame, bg="#ECEFF1")
        self.gray_canvas.pack(fill=tk.BOTH, expand=True)

        binary_frame = tk.LabelFrame(image_frame, text="Бинарное")
        binary_frame.grid(row=0, column=2, sticky="nsew", padx=(4, 0), pady=4)
        self.binary_canvas = tk.Canvas(binary_frame, bg="#ECEFF1")
        self.binary_canvas.pack(fill=tk.BOTH, expand=True)

        hist_frame = tk.LabelFrame(self.root, text="Гистограмма яркости", padx=8, pady=8)
        hist_frame.pack(fill=tk.BOTH, padx=10, pady=(0, 10))

        self.fig, self.ax = plt.subplots(figsize=(8, 2.8))
        self.canvas_hist = FigureCanvasTkAgg(self.fig, master=hist_frame)
        self.canvas_hist.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.original_canvas.bind("<Configure>", self.on_resize)
        self.gray_canvas.bind("<Configure>", self.on_resize)
        self.binary_canvas.bind("<Configure>", self.on_resize)

        self.set_ui_state(False)

    def set_ui_state(self, has_image):
        state = tk.NORMAL if has_image else tk.DISABLED
        self.threshold_slider.config(state=state)
        self.threshold_spinbox.config(state=state)
        self.apply_manual_btn.config(state=state)
        self.otsu_btn.config(state=state)
        self.save_btn.config(state=state if self.binary_image is not None else tk.DISABLED)

    def set_calculation_report(self, text):
        self.calc_text.config(state=tk.NORMAL)
        self.calc_text.delete("1.0", tk.END)
        self.calc_text.insert("1.0", text)
        self.calc_text.config(state=tk.DISABLED)

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All files", "*.*")],
        )
        if not file_path:
            return

        try:
            pil_img = Image.open(file_path)
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")

            self.original_image = np.array(pil_img)
            self.gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)
            self.binary_image = None
            self.current_threshold = 128
            self.last_method = "-"

            self.compute_histogram()

            self.display_image(self.original_image, self.original_canvas, "original")
            self.display_image(self.gray_image, self.gray_canvas, "gray")
            self.binary_canvas.delete("all")
            self.photo_refs["binary"] = None

            self.file_label.config(text=os.path.basename(file_path), fg="black")
            self.manual_result_text.set("Ручной порог: -")
            self.otsu_result_text.set("Оптимальный порог (Оцу): -")

            self.update_histogram_plot()
            self.update_stats(method_name="Данные загружены", threshold=None)
            self.set_calculation_report("Отчет появится после запуска метода Оцу.")

            self.set_ui_state(True)
            self.save_btn.config(state=tk.DISABLED)
        except (OSError, ValueError) as e:
            messagebox.showerror("Ошибка загрузки", f"Не удалось загрузить изображение:\n{str(e)}")
            self.original_image = None
            self.gray_image = None
            self.binary_image = None
            self.file_label.config(text="Ошибка загрузки", fg="red")
            self.stats_text.set("Ошибка при чтении изображения.")
            self.set_calculation_report("Ошибка при чтении изображения.")
            self.set_ui_state(False)

    def compute_histogram(self):
        if self.gray_image is None:
            self.histogram = None
            return
        hist = cv2.calcHist([self.gray_image], [0], None, [256], [0, 256])
        self.histogram = hist.flatten()

    def display_image(self, img, canvas, tag):
        if img is None:
            canvas.delete("all")
            return

        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width = img.shape[1]
            canvas_height = img.shape[0]

        img_h, img_w = img.shape[:2]
        scale = min(canvas_width / img_w, canvas_height / img_h)
        new_w = max(1, int(img_w * scale))
        new_h = max(1, int(img_h * scale))

        if len(img.shape) == 2:
            display_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            display_img = img

        pil_img = Image.fromarray(display_img)
        pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(pil_img)
        self.photo_refs[tag] = photo

        canvas.delete("all")
        x = (canvas_width - new_w) // 2
        y = (canvas_height - new_h) // 2
        canvas.create_image(x, y, anchor=tk.NW, image=photo)

    def on_resize(self, _event):
        if self.original_image is not None:
            self.display_image(self.original_image, self.original_canvas, "original")
        if self.gray_image is not None:
            self.display_image(self.gray_image, self.gray_canvas, "gray")
        if self.binary_image is not None:
            self.display_image(self.binary_image, self.binary_canvas, "binary")

    def update_histogram_plot(self, threshold=None, label=None):
        if self.histogram is None:
            return

        self.ax.clear()
        x = np.arange(256)
        self.ax.bar(x, self.histogram, width=1.0, color="#78909C", edgecolor="#37474F", alpha=0.85)
        self.ax.set_xlim([0, 255])
        self.ax.set_xlabel("Яркость")
        self.ax.set_ylabel("Количество пикселей")

        if threshold is not None:
            legend_label = label if label is not None else f"t = {threshold}"
            self.ax.axvline(x=threshold, color="#D32F2F", linestyle="--", linewidth=2, label=legend_label)
            self.ax.legend(loc="upper right")

        self.fig.tight_layout()
        self.canvas_hist.draw()

    def on_threshold_change(self, _event=None):
        t = int(round(self.threshold_var.get()))
        self.threshold_var.set(t)
        if self.gray_image is not None:
            self.update_histogram_plot(threshold=t, label="Выбранный порог")

    def on_threshold_spin(self):
        t = int(self.threshold_var.get())
        if self.gray_image is not None:
            self.update_histogram_plot(threshold=t, label="Выбранный порог")

    def apply_manual_threshold(self):
        if self.gray_image is None:
            return

        t = int(self.threshold_var.get())
        self.current_threshold = t
        self.last_method = "Ручная бинаризация"

        self.binary_image = np.where(self.gray_image >= t, 255, 0).astype(np.uint8)

        self.display_image(self.binary_image, self.binary_canvas, "binary")
        self.save_btn.config(state=tk.NORMAL)
        self.manual_result_text.set(f"Ручной порог: {t}")
        self.update_histogram_plot(threshold=t, label="Ручной порог")
        self.update_stats(method_name=self.last_method, threshold=t)

    def run_otsu_with_report(self):
        if self.histogram is None:
            raise ValueError("Гистограмма не рассчитана")

        hist = self.histogram.astype(np.float64)
        total_pixels = np.sum(hist)
        if total_pixels == 0:
            raise ValueError("Пустое изображение")

        prob = hist / total_pixels

        q1 = np.cumsum(prob)
        q2 = 1.0 - q1

        cum_sum = np.cumsum(np.arange(256) * prob)
        with np.errstate(divide="ignore", invalid="ignore"):
            M1 = np.where(q1 > 0, cum_sum / q1, 0)
            M2 = np.where(q2 > 0, (cum_sum[-1] - cum_sum) / q2, 0)

        sigma_b_squared = q1 * q2 * (M1 - M2) ** 2

        valid = (q1 > 0) & (q2 > 0)
        sigma_for_choice = sigma_b_squared.copy()
        sigma_for_choice[~valid] = -1

        best_t = int(np.argmax(sigma_for_choice))
        max_sigma = sigma_for_choice[best_t]

        binary_image = np.where(self.gray_image >= best_t, 255, 0).astype(np.uint8)
        black_pixels = int(np.sum(binary_image == 0))
        white_pixels = int(np.sum(binary_image == 255))

        lines = []
        lines.append("\n" + "=" * 80)
        lines.append("                    МЕТОД ОЦУ — АВТОМАТИЧЕСКАЯ БИНАРИЗАЦИЯ")
        lines.append("=" * 80)

        lines.append("\n1. ИНФОРМАЦИЯ ОБ ИЗОБРАЖЕНИИ")
        lines.append(f"   Общее количество пикселей N = {int(total_pixels)}")
        lines.append("   Диапазон яркостей: 0..255")
        lines.append(f"   Размер изображения: {self.gray_image.shape[1]}x{self.gray_image.shape[0]}")

        lines.append("\n2. ПОЛНАЯ ГИСТОГРАММА ЯРКОСТЕЙ")
        lines.append("   Формат: [Яркость] -> Количество пикселей (Вероятность)")
        lines.append("   " + "-" * 50)

        non_zero_count = 0
        for i in range(256):
            if hist[i] > 0:
                lines.append(f"   [{i:3d}] -> {int(hist[i]):8d} пикс. (p = {prob[i]:.6f})")
                non_zero_count += 1

        if non_zero_count == 0:
            lines.append("   ВНИМАНИЕ: Все значения гистограммы равны нулю!")
        else:
            lines.append("   " + "-" * 50)
            lines.append(f"   Всего ненулевых уровней яркости: {non_zero_count}")

        lines.append("\n3. ПОЛНЫЙ ПРОЦЕСС ПОИСКА ОПТИМАЛЬНОГО ПОРОГА")
        lines.append("   Проверяются все t от 0 до 255")
        lines.append("\n   " + "-" * 100)
        lines.append(f"   {'t':>4} | {'q1(t)':>12} | {'q2(t)':>12} | {'M1(t)':>12} | {'M2(t)':>12} | {'σ²_B(t)':>15} | Статус")
        lines.append("   " + "-" * 100)

        valid_count = 0
        for t in range(256):
            if valid[t]:
                status = "✓"
                valid_count += 1
            else:
                if q1[t] == 0:
                    status = "q1=0"
                elif q2[t] == 0:
                    status = "q2=0"
                else:
                    status = "—"

            sigma_str = f"{sigma_for_choice[t]:15.2f}" if valid[t] else "        —"
            lines.append(
                f"   {t:4d} | {q1[t]:12.6f} | {q2[t]:12.6f} | {M1[t]:12.4f} | {M2[t]:12.4f} | {sigma_str} | {status}"
            )

        lines.append("   " + "-" * 100)
        lines.append(f"   Всего валидных порогов (q1>0 и q2>0): {valid_count} из 256")

        sorted_indices = np.argsort(sigma_for_choice)[::-1]
        lines.append("\n4. ТОП-5 ЛУЧШИХ ПОРОГОВ (по убыванию σ²_B):")
        lines.append("   " + "-" * 60)
        lines.append(f"   {'Место':>5} | {'t':>4} | {'σ²_B(t)':>15} | {'q1':>12} | {'q2':>12} | {'M1':>10} | {'M2':>10}")
        lines.append("   " + "-" * 60)

        rank = 0
        for t in sorted_indices:
            if not valid[t]:
                continue
            rank += 1
            lines.append(
                f"   {rank:5d} | {t:4d} | {sigma_for_choice[t]:15.2f} | {q1[t]:12.4f} | {q2[t]:12.4f} | {M1[t]:10.2f} | {M2[t]:10.2f}"
            )
            if rank == 5:
                break

        lines.append("\n5. ОПТИМАЛЬНЫЙ ПОРОГ")
        lines.append("   " + "=" * 40)
        lines.append(f"   t* = {best_t}")
        lines.append(f"   Максимальная межклассовая дисперсия σ²_B(t*) = {max_sigma:.4f}")
        lines.append("\n   Характеристики классов при оптимальном пороге:")
        lines.append("   " + "-" * 40)
        lines.append(f"   Класс 1 (фон):     яркости [0 .. {best_t}]")
        lines.append(f"   Класс 2 (объект):  яркости [{best_t + 1} .. 255]")
        lines.append(f"\n   q1({best_t}) = {q1[best_t]:.6f} (доля пикселей в классе 1)")
        lines.append(f"   q2({best_t}) = {q2[best_t]:.6f} (доля пикселей в классе 2)")
        lines.append(f"\n   M1({best_t}) = {M1[best_t]:.4f} (средняя яркость класса 1)")
        lines.append(f"   M2({best_t}) = {M2[best_t]:.4f} (средняя яркость класса 2)")
        lines.append(f"\n   Проверка: q1 + q2 = {q1[best_t] + q2[best_t]:.6f} (должно быть 1.0)")

        lines.append("\n6. РЕЗУЛЬТАТ БИНАРИЗАЦИИ")
        lines.append("   " + "-" * 40)
        lines.append("   Правило преобразования:")
        lines.append(f"   если яркость < {best_t}  -> 0 (черный)")
        lines.append(f"   если яркость >= {best_t} -> 255 (белый)")
        lines.append("\n   После бинаризации:")
        lines.append(f"   Черных пикселей (0):   {black_pixels:8d} ({100 * black_pixels / total_pixels:.2f}%)")
        lines.append(f"   Белых пикселей (255):  {white_pixels:8d} ({100 * white_pixels / total_pixels:.2f}%)")
        lines.append(f"   Всего пикселей:        {int(total_pixels):8d} (100.00%)")
        lines.append("=" * 80 + "\n")

        return best_t, max_sigma, binary_image, "\n".join(lines)

    def apply_otsu(self):
        if self.gray_image is None:
            return

        try:
            best_t, max_sigma, binary_image, report_text = self.run_otsu_with_report()
        except ValueError as e:
            messagebox.showerror("Ошибка", str(e))
            return

        self.current_threshold = best_t
        self.last_method = "Метод Оцу"
        self.threshold_var.set(best_t)
        self.binary_image = binary_image

        self.display_image(self.binary_image, self.binary_canvas, "binary")
        self.save_btn.config(state=tk.NORMAL)

        self.otsu_result_text.set(f"Оптимальный порог (Оцу): {best_t}, σ²_B = {max_sigma:.4f}")
        self.update_histogram_plot(threshold=best_t, label="Порог Оцу")
        self.update_stats(method_name=self.last_method, threshold=best_t)
        self.set_calculation_report(report_text)

    def update_stats(self, method_name, threshold):
        if self.gray_image is None:
            self.stats_text.set("Нет данных")
            return

        h, w = self.gray_image.shape[:2]
        total = h * w

        lines = [
            f"Размер: {w}x{h} ({total} пикс.)",
            f"Средняя яркость: {self.gray_image.mean():.2f}",
            f"Ст. отклонение яркости: {self.gray_image.std():.2f}",
            f"Метод: {method_name}",
            f"Порог: {threshold if threshold is not None else '-'}",
        ]

        if self.binary_image is not None:
            black = int(np.sum(self.binary_image == 0))
            white = int(np.sum(self.binary_image == 255))
            lines.extend(
                [
                    "",
                    "Результат бинаризации:",
                    f"Черных: {black} ({100.0 * black / total:.2f}%)",
                    f"Белых:  {white} ({100.0 * white / total:.2f}%)",
                ]
            )

        self.stats_text.set("\n".join(lines))

    def save_image(self):
        if self.binary_image is None:
            return

        file_path = filedialog.asksaveasfilename(
            title="Сохранить бинарное изображение",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")],
        )
        if not file_path:
            return

        try:
            Image.fromarray(self.binary_image).save(file_path)
            messagebox.showinfo("Сохранение", f"Изображение сохранено:\n{file_path}")
        except (OSError, ValueError) as e:
            messagebox.showerror("Ошибка сохранения", str(e))
