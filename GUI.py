import tkinter as tk
from functools import partial
import lin, knn, ridge, ridgeCV, sgd, deleteFiles, graphresult
import os


SAVEIMG = False
SAVEM = False
SAMPLE = 25
ROWS = 100000
DATA = "datasets/2015_data.csv"

#function to change which year data is from
def change_year(x):
    global DATA
    year = { 1: "Current Year: 2005", 2: "Current Year: 2006",
               3: "Current Year: 2007", 4: "Current Year: 2008",
               5: "Current Year: 2009", 6: "Current Year: 2010",
               7: "Current Year: 2011", 8: "Current Year: 2012",
               9: "Current Year: 2013", 10: "Current Year: 2014",
               11: "Current Year: 2015"}
    switch = { 1: "datasets/2005_data.csv", 2: "datasets/2006_data.csv",
               3: "datasets/2007_data.csv", 4: "datasets/2008_data.csv",
               5: "datasets/2009_data.csv", 6: "datasets/2010_data.csv",
               7: "datasets/2011_data.csv", 8: "datasets/2012_data.csv",
               9: "datasets/2013_data.csv", 10: "datasets/2014_data.csv",
               11: "datasets/2015_data.csv"}
    year_var.set(year.get(x))
    DATA = switch.get(x, "datasets/2015_data.csv")


#function to change whether graph is saved or not
def save_image():
    global SAVEIMG
    SAVEIMG = not SAVEIMG
    if not SAVEIMG:
        save_img_text.set("Save Image = False")
    else:
        save_img_text.set("Save Image = True")
    
    

#function to change whether machine is saved or not
def save_machine():
    global SAVEM
    SAVEM = not SAVEM
    if not SAVEM:
        save_machine_text.set("Save Machine = False")
    else:
        save_machine_text.set("Save Machine = True")

def del_saves(x):
    deleteFiles.delete_contents(x)
    

#function to run different regresion alogrithms
def run_regressor(x):
    global SAVEIMG, SAVEM, SAMPLE, ROWS, DATA

    #changes sample size
    if len(sample_display.get()) >0:
        sample_new = int(sample_display.get())

        if sample_new > 0 and sample_new < 40:
            SAMPLE = sample_new
        else:
            SAMPLE = 25

    #changes algorithm
    if x == 0:
        lin.linear_regressor(DATA, ROWS, SAMPLE, saveImg=SAVEIMG, saveM=SAVEM)
    if x == 1:
        knn.knn_regressor(DATA, ROWS, SAMPLE, saveImg=SAVEIMG, saveM=SAVEM)
    if x == 2:
        ridge.ridge_regressor(DATA, ROWS, SAMPLE, saveImg=SAVEIMG, saveM=SAVEM)
    if x == 3:
        ridgeCV.ridgeCV_regressor(DATA, ROWS, SAMPLE, saveImg=SAVEIMG, saveM=SAVEM)
    if x == 4:
        sgd.sgd_regressor(DATA, ROWS, SAMPLE, saveImg=SAVEIMG, saveM=SAVEM)

def display_images():
    graphresult.graph_all()
    


root = tk.Tk()
root.title("Life Expectancy Predictor")

frame = tk.Frame(master=root)
frame.pack()

#labels for each of the columns of buttons
year_var = tk.StringVar()
year_label = tk.Label(frame, textvariable=year_var)
year_var.set("Current Year: 2015")
year_label.grid(row=0, column=0)

save_label = tk.Label(frame, text="Save Options")
save_label.grid(row=0, column=2)

algo_label = tk.Label(frame, text="Algorithms")
algo_label.grid(row=0, column=4)


#Create buttons to change year of data
for i in range (12):
    if i > 0:
        year = str(i+2004)
        yearBtn = tk.Button(frame, text=year, width= 9, command=partial(change_year, i))
        yearBtn.grid(row=i, column=0)


#Make space between button categories
for i in range (12):
    f = tk.Frame(master=frame , width=30)
    f.grid(row=i, column=1)
for i in range (12):
    f = tk.Frame(master=frame , width=30)
    f.grid(row=i, column=3)
    
    

#Button to save image or not
save_img_text = tk.StringVar()
Saveimg = tk.Button(frame, textvariable=save_img_text, width= 17, command=save_image)
save_img_text.set("Save Image = False")
Saveimg.grid(row=1, column=2)

#button to save machine or not
save_machine_text = tk.StringVar()
Savemachine = tk.Button(frame, textvariable=save_machine_text, width= 17, command=save_machine)
save_machine_text.set("Save Machine = False")
Savemachine.grid(row=2, column=2)

#buttons for each regression algorithm
lin_re = tk.Button(frame, text="Linear Regressor", width=15, command=partial(run_regressor, 0))
lin_re.grid(row=1, column=4)

knn_reg = tk.Button(frame, text="KNN Regressor", width=15, command=partial(run_regressor, 1))
knn_reg.grid(row=2, column=4)

ridge_reg = tk.Button(frame, text="Ridge Regressor", width=15, command=partial(run_regressor, 2))
ridge_reg.grid(row=3, column=4)

ridgeCV_reg = tk.Button(frame, text="RidgeCV Regressor", width=15, command=partial(run_regressor, 3))
ridgeCV_reg.grid(row=4, column=4)

sgp_reg = tk.Button(frame, text="SGP Regressor", width=15, command=partial(run_regressor, 4))
sgp_reg.grid(row=5, column=4)

#provides ability to set sample size of graphs
sample_label = tk.Label(frame, text="Graph Display Size (int < 40 only!)")
sample_label.grid(row=4, column=2)

sample_display = tk.Entry(frame)
sample_display.grid(row=5, column=2)


#provide ability to delete saved graphs and machines

del_images_btn = tk.Button(frame, text="Delete Saved Images", width=17, command=partial(del_saves, 1))
del_images_btn.grid(row=7, column=2)

del_machine_btn = tk.Button(frame, text="Delete Saved Machines", width=17, command=partial(del_saves, 0))
del_machine_btn.grid(row=8, column=2)

display_saved_btn = tk.Button(frame, text="Display Saved Images", width=15, command=display_images)
display_saved_btn.grid(row=7, column=4)


root.mainloop()