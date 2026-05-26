import tkinter as tk
from PIL import Image, ImageDraw
import cv2
import numpy as np
import os
from keras.models import load_model

CLASSES = ['circle', 'square', 'triangle', 'star']
MODEL_PATH = 'my_shape_model.h5'

model = load_model(MODEL_PATH)

class ShapeDrawingApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Ứng dụng Vẽ & Nhận Diện Hình Học")
        self.window.geometry("400x520")
        self.window.resizable(False, False)

        self.title_label = tk.Label(window, text="BẢNG VẼ HÌNH HỌC AI (CNN)", font=("Arial", 14, "bold"))
        self.title_label.pack(pady=5)

        self.canvas = tk.Canvas(window, width=320, height=320, bg='black', cursor="pencil")
        self.canvas.pack(pady=5)

        self.image_buffer = Image.new("RGB", (320, 320), "black")
        self.draw_context = ImageDraw.Draw(self.image_buffer)

        self.canvas.bind("<B1-Motion>", self.draw_line)

        self.result_label = tk.Label(window, text="Hãy vẽ một hình...", font=("Arial", 11, "italic"), fg="blue")
        self.result_label.pack(pady=10)

        button_frame = tk.Frame(window)
        button_frame.pack(pady=5)

        self.btn_predict = tk.Button(button_frame, text="🧠 AI Nhận Diện", font=("Arial", 11, "bold"), bg="#28a745", fg="white", command=self.predict_shape)
        self.btn_predict.pack(side=tk.LEFT, padx=15)

        self.btn_clear = tk.Button(button_frame, text="🗑️ Xóa Bảng Vẽ", font=("Arial", 11), bg="#dc3545", fg="white", command=self.clear_board)
        self.btn_clear.pack(side=tk.RIGHT, padx=15)

    def draw_line(self, event):
        brush_size = 6 # 🌟 Đã giảm từ 10 xuống 6 để nét vẽ thanh mảnh, giống bộ dữ liệu mẫu hơn
        x1, y1 = (event.x - brush_size), (event.y - brush_size)
        x2, y2 = (event.x + brush_size), (event.y + brush_size)
        self.canvas.create_oval(x1, y1, x2, y2, fill='white', outline='white')
        self.draw_context.ellipse([x1, y1, x2, y2], fill='white', outline='white')

    def clear_board(self):
        self.canvas.delete("all")
        self.image_buffer = Image.new("RGB", (320, 320), "black")
        self.draw_context = ImageDraw.Draw(self.image_buffer)
        self.result_label.config(text="Hãy vẽ một hình...", fg="blue", font=("Arial", 11, "italic"))

    def predict_shape(self):
        img_np = np.array(self.image_buffer)
        img_resized = cv2.resize(img_np, (32, 32))
        
        # 🌟 KHÁC BIỆT: Giữ nguyên ma trận vuông (32, 32, 3) và thêm chiều batch thành (1, 32, 32, 3) cho mạng CNN
        img_ready = np.expand_dims(img_resized, axis=0).astype('float32') / 255.0
        
        predictions = model.predict(img_ready)
        best_match_idx = np.argmax(predictions)
        
        predicted_name = CLASSES[best_match_idx]
        confidence_percent = predictions[0][best_match_idx] * 100
        
        self.result_label.config(
            text=f"AI Dự Đoán: {predicted_name.upper()} ({confidence_percent:.2f}%)",
            font=("Arial", 12, "bold"),
            fg="#28a745" if confidence_percent > 70 else "#ffc107"
        )

if __name__ == "__main__":
    root = tk.Tk()
    app = ShapeDrawingApp(root)
    root.mainloop()