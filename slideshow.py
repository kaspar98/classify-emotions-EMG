import time
import pandas as pd
import tkinter as tk

from itertools import cycle
from PIL.ImageTk import PhotoImage


GROUPS = ["happy", "neutral", "sad"]
SAMPLE_SIZE = 10


def get_images():
    legend = pd.read_csv("legend.csv") 

    retval = []
    for group in GROUPS:
        retval.extend(legend[legend["emotion"] == group].sample(SAMPLE_SIZE)["Theme"])

    return list(map(lambda item: "img/" + item + ".jpg", retval))


class Imagewindow(tk.Tk):

    def __init__(self, images):
        tk.Tk.__init__(self)

        self.count = len(images)
        self.index = 0

        self.blank_shown = False
        
        self.blank = PhotoImage(file="img/blank.jpg")
        self.photos = cycle(PhotoImage(file=image) for image in images)
        self.displayCanvas = tk.Label(self)
        self.displayCanvas.pack()


    def slideShow(self):
        if (self.index >= self.count):
            self.quit()

        if self.index % SAMPLE_SIZE == 0 and not self.blank_shown:
            self.blank_shown  = True

            self.displayCanvas.config(image=self.blank)
            self.after(1000 * 5, self.slideShow)
        else:
            self.index += 1
            self.blank_shown = False

            img = next(self.photos)
            self.displayCanvas.config(image=img)
            self.after(1000 * 2, self.slideShow)


    def run(self):
        self.mainloop()


imagelist = get_images()

root = Imagewindow(images=imagelist)
root.overrideredirect(True)
root.geometry('500x400')
root.slideShow()
root.run()