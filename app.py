import cv2
import time
import tkinter as tk
from PIL import Image, ImageTk
from ultralytics import YOLO

SEGUNDOS_DE_CAPTURA = 30

class YOLOApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        
        # Cargar el modelo YOLO (yolov8n es la versión nano, ideal para tiempo real)
        # La primera vez que lo corras, descargará el archivo .pt automáticamente.
        self.model = YOLO('yolov8n.pt')
        
        self.video_source = 0
        self.vid = None
        self.is_running = False
        self.start_time = 0
        
        # Etiqueta para mostrar el video
        self.video_label = tk.Label(window)
        self.video_label.pack(padx=10, pady=10)
        
        # Etiqueta para el contador
        self.countdown_label = tk.Label(window, text=f"Presiona Iniciar para capturar ({SEGUNDOS_DE_CAPTURA}s)", font=("Helvetica", 16))
        self.countdown_label.pack(pady=10)
        
        # Botón para iniciar
        self.btn_start = tk.Button(window, text="Iniciar Detección", width=20, height=2, command=self.start_capture, font=("Helvetica", 12))
        self.btn_start.pack(pady=10)
        
        # Evento para cerrar la ventana limpiamente
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def start_capture(self):
        if not self.is_running:
            self.vid = cv2.VideoCapture(self.video_source)
            self.is_running = True
            self.start_time = time.time()
            self.btn_start.config(state=tk.DISABLED)
            self.update_frame()
            
    def update_frame(self):
        if not self.is_running:
            return
            
        ret, frame = self.vid.read()
        if not ret:
            self.countdown_label.config(text="Error: No se pudo leer la cámara.")
            self.btn_start.config(state=tk.NORMAL)
            self.is_running = False
            return
            
        # Ejecutar inferencia YOLO filtrando por la clase 0 (Persona en el dataset COCO)
        results = self.model(frame, classes=[0], verbose=False)
        
        # Dibujar las cajas de los resultados en la imagen
        annotated_frame = results[0].plot()
        
        # Calcular tiempo transcurrido y restante
        elapsed_time = time.time() - self.start_time
        remaining_time = max(0, SEGUNDOS_DE_CAPTURA - int(elapsed_time))
        
        self.countdown_label.config(text=f"Tiempo restante: {remaining_time}s")
        
        # Convertir imagen (OpenCV usa BGR, Tkinter usa RGB)
        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        
        # Redimensionar la imagen para hacerla más grande en la interfaz
        # Se calcula el alto manteniendo la proporción (puedes cambiar target_width a 1024 o más)
        target_width = 960
        aspect_ratio = img.height / img.width
        target_height = int(target_width * aspect_ratio)
        img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
        
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
        
        if remaining_time > 0:
            # Llamar de nuevo a esta función después de 15ms
            self.window.after(15, self.update_frame)
        else:
            # Si el tiempo llega a 0, guardamos y detenemos
            self.save_and_stop(annotated_frame)
                
    def save_and_stop(self, frame):
        self.is_running = False
        if self.vid and self.vid.isOpened():
            self.vid.release()
        
        # Guardar la imagen con un sello de tiempo único
        filename = f"data/captura_yolo_{int(time.time())}.jpg"
        cv2.imwrite(filename, frame)
        
        self.countdown_label.config(text=f"¡Imagen guardada como {filename}!")
        self.btn_start.config(state=tk.NORMAL)
        
    def on_closing(self):
        self.is_running = False
        if self.vid and self.vid.isOpened():
            self.vid.release()
        self.window.destroy()

if __name__ == '__main__':
    root = tk.Tk()
    app = YOLOApp(root, "Detección de Humanos - YOLOv8")
    root.mainloop()