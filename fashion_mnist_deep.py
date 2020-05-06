import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv
import tkinter as tk
from PIL import Image, ImageTk

file = "fashion-mnist_train.csv"

fields = []
rows = []
with open(file, 'r') as dataset:
    lines = csv.reader(dataset)
    fields = next(lines)
    for row in lines:
        rows.append(row)
rows = np.array(rows)
x_train, y_train = np.float32(rows[:,1:])/255,rows[:,0]
x_valid, y_valid = x_train[:5000], y_train[:5000]
x_train, y_train = x_train[5000:], y_train[5000:]
print(y_train[0:10])
print(x_train[0:10])
print(x_train[0].reshape(28,28).shape)
#loss: 0.2519 - accuracy: 0.9057 - val_loss: 0.2991 - val_accuracy: 0.8922
#Test accuracy: 0.8938999772071838

file = "fashion-mnist_test.csv"

fields = []
rows = []
with open(file, 'r') as dataset:
    lines = csv.reader(dataset)
    fields = next(lines)
    for row in lines:
        rows.append(row)
rows = np.array(rows)
x_test,y_test = np.float32(rows[:,1:])/255,rows[:,0]

# preprocessing
x_train = x_train.reshape(x_train.shape[0],28,28,1)
x_valid = x_valid.reshape(x_valid.shape[0],28,28,1)
x_test = x_test.reshape(x_test.shape[0],28,28,1)

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_valid = tf.keras.utils.to_categorical(y_valid, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

fashion_mnist_labels = ["T-shirt/top",  # index 0
                        "Trouser",      # index 1
                        "Pullover",     # index 2
                        "Dress",        # index 3
                        "Coat",         # index 4
                        "Sandal",       # index 5
                        "Shirt",        # index 6
                        "Sneaker",      # index 7
                        "Bag",          # index 8
                        "Ankle boot"]   # index 9

model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28,1)))
model.add(tf.keras.layers.Dense(1024, activation='relu'))
model.add(tf.keras.layers.Dropout(.1))
model.add(tf.keras.layers.Dense(1024, activation='relu'))
model.add(tf.keras.layers.Dropout(.1))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))
# Take a look at the model summary
model.summary()


model.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])

model.fit(x_train,
         y_train,
         batch_size=64,
         epochs=10,
         validation_data=(x_valid, y_valid))

# Evaluate the model on test set
score = model.evaluate(x_test, y_test, verbose=0)
# Print test accuracy
print('\n', 'Test accuracy:', score[1])



# displays confusion matrix
y_hat = model.predict(x_test)
print("y_test shape",y_test.shape)
print("y_hat shape",y_hat.shape)
confusion = tf.math.confusion_matrix(np.argmax(y_test,1),np.argmax(y_hat,1))
plt.imshow(np.float32(confusion),cmap=plt.cm.Greys)
plt.show()


# displays list of first 20 test samples from least accurate class
# then displays the first 20 of that class that were guessed incorrectly
confusion_diag = np.diagonal(confusion)
most_confused = np.argmin(confusion_diag)
confused_indices = [i if np.argmax(x)==most_confused else None for i,x in enumerate(y_test)]
confused_indices = np.array(confused_indices)
confused_indices = confused_indices[confused_indices!=None]
print(confused_indices[:20])
doesnt_know = [x if np.argmax(y_hat[x])!=most_confused else None for x in confused_indices]
doesnt_know = np.array(doesnt_know)
doesnt_know = doesnt_know[doesnt_know!=None]
print(doesnt_know[:20])



# interactive testing/UI
window = tk.Tk()
window.title("Interactive Testing")
lbl = tk.Label(window, text="Pick a test sample (0-9,999)")
lbl.grid(column=0, row=0)
spin = tk.Spinbox(window, from_=0, to=9999, width=5)
spin.grid(column=1, row=0)
def clicked():
    res = spin.get()
    try:
        sampleId = int(res)
        if sampleId>=0 and sampleId<10000:
            predicted = np.argmax(y_hat[sampleId])
            actual = np.argmax(y_test[sampleId])
            str = "Sample " + res + ": Actual: " + fashion_mnist_labels[actual] + " Predicted: " + fashion_mnist_labels[predicted]
            if predicted==actual:
                bgColor = "green"
            else:
                bgColor = "red"
            lbl2.configure(text=str, bg=bgColor)
            a = x_test[sampleId].reshape(28,28)*255
            img =  ImageTk.PhotoImage(image=Image.fromarray(a))
            img = img._PhotoImage__photo.zoom(10,10)
            canvas.img = img
            canvas.itemconfig(canvas.create_image(3,3, anchor="nw", image=img), image=img)
        else:
            lbl2.configure(text="")
    except:
        lbl2.configure(text="")
btn = tk.Button(window, text="Test", command=clicked)
btn.grid(column=2, row=0)
predicted = np.argmax(y_hat[0])
actual = np.argmax(y_test[0])
str = "Sample " + "0" + ": Actual: " + fashion_mnist_labels[actual] + " Predicted: " + fashion_mnist_labels[predicted]
if predicted==actual:
    bgColor = "green"
else:
    bgColor = "red"
lbl2 = tk.Label(window, text=str, bg=bgColor)
lbl2.grid(column=0, row=1)
a = x_test[0].reshape(28,28)*255
img =  ImageTk.PhotoImage(image=Image.fromarray(a))
img = img._PhotoImage__photo.zoom(10,10)
canvas = tk.Canvas(window,width=280,height=280)
canvas.grid(column=0,row=2)
canvas.create_image(3,3, anchor="nw", image=img)
window.mainloop()



# change done to False to run without tkinter and PIL
done = True
while not done:
    print("Type a number (0-9,999): ",end="")
    try:
        i = int(input())
    except:
        print("Done")
        done = True
        continue
    if i<0 or i>9999:
        print("Try again")
        continue
    plt.imshow(np.squeeze(x_test[i]))
    predict_index = np.argmax(y_hat[i])
    true_index = np.argmax(y_test[i])
    # Set the title for each image
    plt.title("{} ({})".format(fashion_mnist_labels[predict_index],
                                  fashion_mnist_labels[true_index]),
                                  color=("green" if predict_index == true_index else "red"))
    plt.show()
