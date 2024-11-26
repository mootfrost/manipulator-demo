import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image, ImageTk
import torch.nn.functional as F
import torch.nn as nn
import cv2
from tkinter import Tk, Label, Button
import numpy as np

CLASSES = ['Defective', 'Normal']

def load_model(model_path):
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

model = load_model("grob.pth")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict(image):
    """Функция для предсказания класса изображения."""
    image = transform(image).unsqueeze(0)  
    with torch.no_grad():
        outputs = model(image)
        probabilities = F.softmax(outputs[0], dim=0)
        _, predicted_idx = torch.max(probabilities, dim=0)
        predicted_class = CLASSES[predicted_idx % len(CLASSES)]
        return predicted_class

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("ГРОБ ИИ")

        self.video_label = Label(root)
        self.video_label.pack()

        self.capture_button = Button(root, text="Capture", command=self.capture_image)
        self.capture_button.pack()

        self.result_label = Label(root, text="", font=("Arial", 16))
        self.result_label.pack()

        self.cap = cv2.VideoCapture(0)
        self.update_frame()

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        self.root.after(10, self.update_frame)

    def capture_image(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            predicted_class = predict(image)
            self.result_label.config(text=f"Class: {predicted_class}")

    def __del__(self):
        self.cap.release()

if __name__ == "__main__":
    root = Tk()
    app = App(root)
    root.mainloop()
