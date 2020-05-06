from tkinter import *
import finalProjectLR as LR
import finalProjectNN as NN
import finalProjectSVM as SVM
from PIL import Image, ImageTk


window = Tk()

window.title("Comparing Methods of Detecting Credit Fraud")
#window.attributes('-fullscreen', True)
title = Label(window, text="Comparing Methods of Detecting Credit Fraud", font=("Arial Black", 30))
#window.geometry('800x600')
title.grid(column=0, row=0, columnspan=3, pady = 10)

def click_lr():
    current_model.configure(text="Current Model: Logistic Regression")
    retlist = LR.run_LR()
    configure_results(retlist)
    update_images()

def click_nn():
    current_model.configure(text="Current Model: Neural Network")
    retlist = NN.run_NN()
    configure_results(retlist)
    update_images()

def click_svm():
    current_model.configure(text="Current Model: SVM")
    retlist = SVM.run_SVM()
    configure_results(retlist)
    update_images()

lrbtn = Button(window, text="Run Logistic Regression", bg="black", fg="white", command=click_lr, font=("Arial Black", 15))
lrbtn.grid(column=0, row=1, pady = 10)
nnbtn = Button(window, text="Run Neural Network Model", bg="black", fg="white", command=click_nn, font=("Arial Black", 15))
nnbtn.grid(column=1, row=1, pady = 10)
svmbtn = Button(window, text="Run SVM Model", bg="black", fg="white", command=click_svm, font=("Arial Black", 15))
svmbtn.grid(column=2, row=1, pady = 10)


current_model = Label(window, text="Current Model: Not yet run", font=("Arial Black", 20))
current_model.grid(column=0, row=2, columnspan=3, pady = 10)

n = Label(window, text="n = ?", font=("Arial Black", 10),relief="groove")
n.grid(column=0, row=3, pady = 5)
accuracy = Label(window, text="Accuracy = ?", font=("Arial Black", 10),relief="groove")
accuracy.grid(column=1, row=3, pady = 5)
precision = Label(window, text="Precision = ?", font=("Arial Black", 10),relief="groove")
precision.grid(column=2, row=3, pady = 5)
recall = Label(window, text="Recall = ?", font=("Arial Black", 10), relief="groove")
recall.grid(column=0, row=4, columnspan = 2, pady = 5)
f1_score = Label(window, text="F1 Score = ?", font=("Arial Black", 10), relief="groove")
f1_score.grid(column=1, row=4, columnspan = 2, pady = 5)

load = Image.open("placeholder.png")
render = ImageTk.PhotoImage(load)
img1 = Label(window, image=render,)
img1.image = render
img1.grid(column=0, row = 5, padx = 5, pady = 5)
img2 = Label(window, image=render)
img2.image = render
img2.grid(column=1, row = 5, padx = 5, pady = 5)
img3 = Label(window, image=render)
img3.image = render
img3.grid(column=2, row = 5, padx =5, pady = 5)

def update_images():
    load = Image.open("actual_vs_predicted.png")
    render = ImageTk.PhotoImage(load)
    img1.configure(image=render)
    img1.image = render
    load = Image.open("false_true_rates.png")
    render = ImageTk.PhotoImage(load)
    img2.configure(image=render)
    img2.image = render
    load = Image.open("loss_vs_epoch.png")
    render = ImageTk.PhotoImage(load)
    img3.configure(image=render)
    img3.image = render

def configure_results(retlist):
    n.configure(text=retlist[0])
    accuracy.configure(text=retlist[1])
    retlist[2]=retlist[2].replace("[","")
    retlist[2]=retlist[2].replace("]","")
    retlist[3]=retlist[3].replace("[","")
    retlist[3]=retlist[3].replace("]","")
    retlist[4]=retlist[4].replace("[","")
    retlist[4]=retlist[4].replace("]","")
    precision.configure(text=retlist[2])
    recall.configure(text=retlist[3])
    f1_score.configure(text=retlist[4])


exitbtn = Button(window, text="Close Program", bg="red", fg="black", command=exit)
exitbtn.grid(column=1, row=6, pady=20)

def click_lr():
    current_model.configure(text="Current Model: Logistic Regression")

def click_nn():
    current_model.configure(text="Current Model: Neural Network")

def click_svm():
    current_model.configure(text="Current Model: SVM")

window.mainloop()
