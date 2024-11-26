import torch
import torchvision.transforms as transforms
from PIL import Image, ImageTk
import torch.nn.functional as F
import cv2
import torch.nn as nn
from torchvision import models
from tkinter import Tk, Label, Button, StringVar, OptionMenu, ttk
import serial
import serial.tools.list_ports
import time
import threading


CLASSES = ["Defective", "Normal"]

def load_model(model_path, num_classes=2):
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model("models/grob.pth", num_classes=2)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict(image):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        probabilities = F.softmax(outputs[0], dim=0)
        _, predicted_idx = torch.max(probabilities, dim=0)
        return CLASSES[predicted_idx]

class App:
    def __init__(self, root):
        self.root = root
        self.main_frame = ttk.Frame(self.root, padding=(20, 20))
        self.main_frame.pack(padx=20, pady=20)
        self.root.title("Defect Detection")
        
        self.video_label = Label(self.main_frame)
        self.video_label.pack()

        self.capture_button = Button(self.main_frame, text="Start Scanning (No Arduino)", command=self.start_scanning_no_arduino)
        self.capture_button.pack()

        self.result_label = Label(self.main_frame, text="", font=("Arial", 16))
        self.result_label.pack()

        self.arduino_button = Button(self.main_frame, text="Start Scanning (With Arduino)", command=self.start_arduino)
        self.arduino_button.pack()
        
        self.stop_arduino_button = Button(self.main_frame, text="Stop Scanning (With Arduino)", command=self.stop_arduino)
        self.stop_arduino_button.pack()

        self.port_label = Label(self.main_frame, text="Select Arduino Port:")
        self.port_label.pack()

        self.port_options = self.get_serial_ports()
        self.selected_port = StringVar(self.main_frame)
        self.selected_port.set(self.port_options[0] if self.port_options else "No Ports Found")
        
        self.port_menu = OptionMenu(self.main_frame, self.selected_port, *self.port_options)
        self.port_menu.pack()

        self.cap = cv2.VideoCapture(2)
        self.update_frame()


        self.serial_connection = None

    def update_frame(self):
        ret, frame = self.cap.read()
        (height, width) = frame.shape[:2]
        frame = frame[0:height, width//2-100:width]
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        self.root.after(10, self.update_frame)

    def capture_image(self):
        ret, frame = self.cap.read()
        (height, width) = frame.shape[:2]
        frame = frame[0:height, width//2-100:width]
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(frame)
        return None
    
    def stop_scanning_with_arduino(self):
        self.running = False
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.close()
            self.result_label.config(text="Stopped scanning with Arduino")
        else:
            self.result_label.config(text="No connection to Arduino")

    def start_scanning_no_arduino(self):
        image = self.capture_image()
        if image is not None:
            prediction = predict(image)
            self.result_label.config(text=f"Prediction: {prediction}")

            if prediction == "Defective":
                self.root.config(bg="red")
                self.root.after(2000, lambda: self.root.config(bg="gray85"))  # Возвращаем фон через 1 секунду
            else:
                self.root.config(bg="green")
                self.root.after(2000, lambda: self.root.config(bg="gray85"))  # Возвращаем фон через 1 секунду


    def get_serial_ports(self):
        ports = serial.tools.list_ports.comports()
        port_info = [f"{port.device} - {port.description}" for port in ports]
        return port_info

    def connect_to_arduino(self):
        selected_port = self.selected_port.get().split(' - ')[0]

        try:
            self.serial_connection = serial.Serial(selected_port, 9600, timeout=1)
            time.sleep(2)
            self.result_label.config(text=f"Connected to Arduino on {selected_port}")
            return True
        except Exception as e:
            self.result_label.config(text=f"Failed to connect: {e}")
            return False

    def send_command(self, command):
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.write(command.encode())
            time.sleep(0.1)

    def read_response(self):
        if self.serial_connection and self.serial_connection.is_open:
            response = self.serial_connection.readline().decode().strip()
            return response
        return ""
    
    def start_arduino(self):
        self.running = True
        self.arduino_thread = threading.Thread(target=self.start_scanning_with_arduino)
        self.arduino_thread.start()
    
    def stop_arduino(self):
        """Остановка работы с Arduino."""
        self.running = False
        if hasattr(self, 'arduino_thread'):
            self.arduino_thread.join()
            self.result_label.config(text="Stopped scanning with Arduino")
        else:
            self.result_label.config(text="No connection to Arduino")

    def start_scanning_with_arduino(self):
        """Запуск процесса сканирования с Arduino."""
        if not self.connect_to_arduino():
            return
        print(self.running)
        while self.running:
            time.sleep(10)
            self.send_command("M")
            while (self.read_response() != "D"):
                pass
            if True:
                image = self.capture_image()
                if image is not None:
                    prediction = predict(image)
                    self.result_label.config(text=f"Prediction: {prediction}")

                    if prediction == "Defective":
                        self.root.config(bg="red")
                        self.root.after(2000, lambda: self.root.config(bg="gray85"))  # Возвращаем стандартный фон через 1 секунду
                    else:
                        self.root.config(bg="green")
                        self.root.after(2000, lambda: self.root.config(bg="gray85"))  # Возвращаем стандартный фон через 1 секунду
                    print('PRED', prediction)
                    if prediction == "Normal":
                        print('defect')
                        self.send_command("R")
                        while (self.read_response() != "D"):
                            pass
                    else:
                        time.sleep(5)
                        print('norm')
            time.sleep(0.5)


    def __del__(self):
        if self.cap.isOpened():


            self.cap.release()
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.close()

if __name__ == "__main__":
    root = Tk()
    app = App(root)
    root.mainloop()