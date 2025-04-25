import tkinter as tk
import numpy as np
from PIL import Image, ImageDraw
import customtkinter
from model import Model

class UI:
    """ A class that creates the user interface.
    """
    def __init__(self, root, model):
        """ Initializes the UI.

        Args:
        - root: The root Tkinter window
        - model: The neural network model used to make predictions
        """
        self.root = root
        self.model = model

        customtkinter.set_appearance_mode("light")

        self.label = customtkinter.CTkLabel(root, text="Number Recognition", font=("Arial", 20))
        self.label2 = customtkinter.CTkLabel(root, text="Draw a digit and get prediction", font=("Arial", 14))
        self.canvas = tk.Canvas(root, width=300, height=300, bg='white')
        self.predict_button = customtkinter.CTkButton(root, text="Recognise", font=("Arial", 12), command=self.predict_number)
        self.clear_button = customtkinter.CTkButton(root, text="Clear", font=("Arial", 12), command=self.clear_canvas)
        self.result_label = customtkinter.CTkLabel(root, text="Prediction: ", font=("Arial", 14))

        self.label.grid(row=0, column=0, columnspan=3, pady=14)
        self.label2.grid(row=1, column=0, columnspan=3)
        self.canvas.grid(row=2, column=0, columnspan=4, pady=10)
        self.predict_button.grid(row=3, column=1, pady=2)
        self.clear_button.grid(row=3, column=2, pady=2)
        self.result_label.grid(row=4, column=1, columnspan=2, pady=10)

        self.image = Image.new("RGB", (280, 280), (0, 0, 0))
        self.draw_image = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.draw)

    def draw(self, event):
        """Draws on the canvas and updates the internal image/image buffer.

        Args:
            event: Tkinter event object that keeps track of the mouse's position
        """
        x = event.x
        y = event.y
        r = 5
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="black")
        self.draw_image.ellipse([x - r, y - r, x + r, y + r], fill="white")

    def clear_canvas(self):
        """ Clears the canvas, creates new empty internal image and sets up new drawing object.
        """
        self.canvas.delete("all")
        self.result_label.configure(text="Prediction: ")
        self.image = Image.new("RGB", (280, 280), (0, 0, 0))
        self.draw_image = ImageDraw.Draw(self.image)

    def predict_number(self):
        """ Reshapes image to correct format (784x1 column vector), runs it through the model
        and shows the model's prediction.
        """
        img = self.image.resize((28, 28))
        img = img.convert('L')
        img = np.array(img)
        img = img.flatten().reshape((784, 1)) / 255

        res, _ = self.model.forward(img)

        res = np.argmax(res[-1])

        self.result_label.configure(text=f"Prediction: {res}")

def main():
    model = Model(input_layer_size=784, hidden_layer_size=128, output_layer_size=10)
    model.load_latest_model()

    window = customtkinter.CTk()
    window.title("Number Recognition")

    ui = UI(window, model)

    window.mainloop()

if __name__ == "__main__":
    main()
