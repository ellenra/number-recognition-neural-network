import tkinter as tk
import numpy as np
from PIL import Image, ImageDraw
from model import Model
import customtkinter

class UI:
    def __init__(self, root, model):
        self.root = root
        self.model = model

        customtkinter.set_appearance_mode("light")

        self.canvas = tk.Canvas(root, width=300, height=300, bg='white')
        self.label = customtkinter.CTkLabel(root, text="Number Recognition")
        self.predict_button = customtkinter.CTkButton(root, text="Recognise", command=self.predict_number)
        self.clear_button = customtkinter.CTkButton(root, text="Clear", command=self.clear_canvas)
        self.result_label = tk.Label(root, text="Prediction: ")
        
        self.canvas.grid(row=1, column=0, columnspan=4, pady=5)
        self.label.grid(row=0, column=0, columnspan=3, pady=5)
        self.predict_button.grid(row=2, column=1, padx=2)
        self.clear_button.grid(row=2, column=2, pady=2)
        self.result_label.grid(row=3, column=1, columnspan=2, pady=10)

        self.image = Image.new("RGB", (280, 280), (0, 0, 0))
        self.draw_image = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.draw)

    def draw(self, event):
        self.x = event.x
        self.y = event.y
        r = 5
        self.canvas.create_oval(self.x-r, self.y-r, self.x+r, self.y+r, fill="black")
        self.draw_image.ellipse([self.x - r, self.y - r, self.x + r, self.y + r], fill="white")
        
    def clear_canvas(self):
        self.canvas.delete("all")
        self.result_label.config(text="Prediction: ")
        self.image = Image.new("RGB", (280, 280), (0, 0, 0))
        self.draw_image = ImageDraw.Draw(self.image)

    def predict_number(self):
        img = self.image.resize((28, 28))
        img = img.convert('L')
        img = np.array(img)
        img = img.flatten().reshape((784, 1)) / 255

        res, _ = self.model.forward(img)

        res = np.argmax(res[-1])

        self.result_label.config(text=f"Prediction: {res}")

def main():
    model = Model(input_layer_size=784, hidden_layer_size=128, output_layer_size=10)
    model.load_latest_model()

    window = customtkinter.CTk()
    window.title("Number Recognition")
    
    ui = UI(window, model)

    window.mainloop()

if __name__ == "__main__":
    main()
