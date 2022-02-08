# importuri
import os
import tkinter as tk
from tkinter import filedialog
from tkinter import *
import numpy
import imutils
import cv2
from PIL import ImageTk, Image
import numpy as np
from keras.models import load_model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#incarcarea modelului
model = load_model('Model.h5')
classes = {
    0: 'Creier fara tumoare',
    1: 'Creier cu tumoare'
}

#prelucrare imagine
def crop_contur(image):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayscale1 = cv2.GaussianBlur(grayscale, (5, 5), 0)
    threshold_image = cv2.threshold(grayscale, 45, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    threshold_image = cv2.erode(threshold_image, None, iterations=2)
    threshold_image = cv2.dilate(threshold_image, None, iterations=2)

    contur = cv2.findContours(threshold_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contur = imutils.grab_contours(contur)
    c = max(contur, key=cv2.contourArea)
    punct_extrm_stanga = tuple(c[c[:, :, 0].argmin()][0])
    punct_extrm_dreapta = tuple(c[c[:, :, 0].argmax()][0])
    punct_extrm_sus = tuple(c[c[:, :, 1].argmin()][0])
    punct_extrm_jos = tuple(c[c[:, :, 1].argmax()][0])

    new_image = grayscale1[punct_extrm_sus[1]:punct_extrm_jos[1], punct_extrm_stanga[0]:punct_extrm_dreapta[0]]
    new_image = cv2.resize(new_image, (28, 28))


    return new_image

#incarcarea imaginii
def upload_image():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return
    uploaded = Image.open(file_path)
    uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
    im = ImageTk.PhotoImage(uploaded)
    sign_image.configure(image=im)
    sign_image.image = im

    show_classify_button(file_path)

#butonul de start
def show_classify_button(file_path):
    classify_btn = Button(top, text="Start", command=lambda: classify(file_path), padx=10, pady=5)
    classify_btn.configure(background="#364156", foreground="white", font=('arial', 10, 'bold'))
    classify_btn.place(relx=0.79, rely=0.46)

#calsificarea imaginii intrate
def classify(file_path):
    image = cv2.imread(file_path)
    image = crop_contur(image)
    image = numpy.expand_dims(image, axis=0)
    image = numpy.array(image)
    image = image.reshape(image.shape[0], 28, 28, 1)
    prediction = np.argmax(model.predict([image]))
    sign = classes[prediction]
    label.configure(foreground='#011638', text=sign)


# initializare GUI
top = tk.Tk()
top.geometry('800x600')
top.title("Recunoastere Tumori Cerebrale")
top.configure(background="#CDCDCD")

# heading
heading = Label(top, text="Recunoastere Tumori Cerebrale", pady=20, font=('arial', 20, 'bold'))
heading.configure(background="#CDCDCD", foreground='#364156')
heading.pack()

# upload buton
upload = Button(top, text="Incarcati imaginea", command=upload_image, padx=10, pady=5)
upload.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
upload.pack(side=BOTTOM, pady=50)

sign_image = Label(top,background='#CDCDCD')
sign_image.pack(side=BOTTOM, expand=True)

label = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
label.pack(side=BOTTOM, expand=True)

top.mainloop()
