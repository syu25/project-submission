import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv
import cv2 as cv
from scipy.signal import convolve2d

file = "fashion-mnist_train.csv"

fields = []
rows = []
with open(file, 'r') as dataset:
    lines = csv.reader(dataset)
    fields = next(lines)
    for row in lines:
        rows.append(row)
rows = np.array(rows)
#print(fields)
#print(rows[:][0])
print(rows[0:2,0])
x_train, y_train = np.float32(rows[:,1:])/255,rows[:,0]
x_valid, y_valid = x_train[:5000], y_train[:5000]
x_train, y_train = x_train[5000:], y_train[5000:]
print(y_train[0:10])
print(x_train[0:10])
print(type(x_train[0]))
print(type(x_train[0].reshape(28,28)))
print(x_train[0].reshape(28,28).shape)
#img = np.uint8(x_train[0].reshape(28,28)*255)
#img2 = np.uint8((x_train[0].reshape(28,28)**2)*255)
#cv.imshow("img",cv.resize(img,(280,280)))
#cv.imshow("img2",cv.resize(img2,(280,280)))
#cv.waitKey(0)
#cv.destroyAllWindows()
#img = np.uint8(x_train[1].reshape(28,28)*255)
#img2 = np.uint8((x_train[1].reshape(28,28)**2)*255)
#cv.imshow("img",cv.resize(img,(280,280)))
#cv.imshow("img2",cv.resize(img2,(280,280)))
#cv.waitKey(0)
#cv.destroyAllWindows()
#img = np.uint8(x_train[2].reshape(28,28)*255)
#img2 = np.uint8((x_train[2].reshape(28,28)**2)*255)
#cv.imshow("img",cv.resize(img,(280,280)))
#cv.imshow("img2",cv.resize(img2,(280,280)))
#cv.waitKey(0)
#cv.destroyAllWindows()

img = np.uint8(x_train[0].reshape(28,28)*255)
img2 = np.uint8((x_train[0].reshape(28,28)-convolve2d(x_train[0].reshape(28,28),np.array([[1,2,1],[2,4,2],[1,2,1]])/16, mode='same', boundary='fill', fillvalue=0))*255)
cv.imshow("img",cv.resize(img,(280,280)))
cv.imshow("img2",cv.resize(img2,(280,280)))
cv.waitKey(0)
cv.destroyAllWindows()

img = np.uint8(x_train[1].reshape(28,28)*255)
img2 = np.uint8((x_train[1].reshape(28,28)-convolve2d(x_train[1].reshape(28,28),np.array([[1,2,1],[2,4,2],[1,2,1]])/16, mode='same', boundary='fill', fillvalue=0))*255)
cv.imshow("img",cv.resize(img,(280,280)))
cv.imshow("img2",cv.resize(img2,(280,280)))
cv.waitKey(0)
cv.destroyAllWindows()

img = np.uint8(x_train[0].reshape(28,28)*255)
img2 = x_train[0].reshape(28,28)-convolve2d(x_train[0].reshape(28,28),np.array([[1,2,1],[2,4,2],[1,2,1]])/16, mode='same', boundary='fill', fillvalue=0)
img2 = np.uint8(img2/np.amax(img2)*255)
cv.imshow("img",cv.resize(img,(280,280)))
cv.imshow("img2",cv.resize(img2,(280,280)))
cv.waitKey(0)
cv.destroyAllWindows()

img = np.uint8(x_train[1].reshape(28,28)*255)
img2 = x_train[1].reshape(28,28)-convolve2d(x_train[1].reshape(28,28),np.array([[1,2,1],[2,4,2],[1,2,1]])/16, mode='same', boundary='fill', fillvalue=0)
img2 = np.uint8(img2/np.amax(img2)*255)
cv.imshow("img",cv.resize(img,(280,280)))
cv.imshow("img2",cv.resize(img2,(280,280)))
cv.waitKey(0)
cv.destroyAllWindows()

#gray = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
#plt.imshow(np.float32(x_train[0].reshape(28,28)),cmap=plt.cm.Greys)
#plt.show()
#pow4: loss: 0.2237 - accuracy: 0.9177 - val_loss: 0.2873 - val_accuracy: 0.9010 Test accuracy: 0.9017999768257141

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


#x_train = x_train**4
#x_valid = x_valid**4
#x_test = x_test**4

print("Reshape 1 Begin")
x_train = x_train.reshape(x_train.shape[0],28,28)
x_valid = x_valid.reshape(x_valid.shape[0],28,28)
x_test = x_test.reshape(x_test.shape[0],28,28)
print("Reshape 1 End")

x_train_cpy = np.empty((x_train.shape[0],2352))
x_valid_cpy = np.empty((x_valid.shape[0],2352))
x_test_cpy = np.empty((x_test.shape[0],2352))

print("Filters Begin")
print(x_train.shape)
for i,elem in enumerate(x_train):
    elem2 = elem-convolve2d(elem.reshape(28,28),np.array([[1,2,1],[2,4,2],[1,2,1]])/16, mode='same', boundary='fill', fillvalue=0)
    elem3 = elem-convolve2d(elem.reshape(28,28),np.array([[1,4,7,4,1],[4,16,26,16,4],[7,26,41,26,7],[4,16,26,16,4],[1,4,7,4,1]])/273, mode='same', boundary='fill', fillvalue=0)
    x_train_cpy[i] = np.append(elem,elem2/np.amax(elem2),elem3/np.amax(elem3))
print(x_train.shape)
for i,elem in enumerate(x_valid):
    elem2 = elem-convolve2d(elem.reshape(28,28),np.array([[1,2,1],[2,4,2],[1,2,1]])/16, mode='same', boundary='fill', fillvalue=0)
    elem3 = elem-convolve2d(elem.reshape(28,28),np.array([[1,4,7,4,1],[4,16,26,16,4],[7,26,41,26,7],[4,16,26,16,4],[1,4,7,4,1]])/273, mode='same', boundary='fill', fillvalue=0)
    x_valid_cpy[i] = np.append(elem,elem2/np.amax(elem2),elem3/np.amax(elem3))
for i,elem in enumerate(x_test):
    elem2 = elem-convolve2d(elem.reshape(28,28),np.array([[1,2,1],[2,4,2],[1,2,1]])/16, mode='same', boundary='fill', fillvalue=0)
    elem3 = elem-convolve2d(elem.reshape(28,28),np.array([[1,4,7,4,1],[4,16,26,16,4],[7,26,41,26,7],[4,16,26,16,4],[1,4,7,4,1]])/273, mode='same', boundary='fill', fillvalue=0)
    x_test_cpy[i] = np.append(elem,elem2/np.amax(elem2),elem3/np.amax(elem3))
print("Filters End")

print("Reshape 2 Begin")
x_train = x_train_cpy.reshape(x_train_cpy.shape[0],84,28,1)
x_valid = x_valid_cpy.reshape(x_valid_cpy.shape[0],84,28,1)
x_test = x_test_cpy.reshape(x_test_cpy.shape[0],84,28,1)
print("Reshape 2 End")

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
#model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3,activation='relu', padding='valid', input_shape=(28,28,1)))
#model.add(tf.keras.layers.LeakyReLU(alpha=.1))
#model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
#model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='valid', activation='relu'))
#model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(.5, input_shape=(84,28,1)))
model.add(tf.keras.layers.Flatten())
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
         validation_data=(x_valid, y_valid))#,
         #callbacks=[checkpointer])

# Evaluate the model on test set
score = model.evaluate(x_test, y_test, verbose=0)
# Print test accuracy
print('\n', 'Test accuracy:', score[1])



y_hat = model.predict(x_test)

print("y_test shape",y_test.shape)
print("y_hat shape",y_hat.shape)
confusion = tf.math.confusion_matrix(np.argmax(y_test,1),np.argmax(y_hat,1))
plt.imshow(np.float32(confusion),cmap=plt.cm.Greys)
plt.show()

# Plot a random sample of 10 test images, their predicted labels and ground truth
figure = plt.figure(figsize=(20, 8))
for i, index in enumerate(np.random.choice(x_test.shape[0], size=15, replace=False)):
    ax = figure.add_subplot(3, 5, i + 1, xticks=[], yticks=[])
    # Display each image
    ax.imshow(np.squeeze(x_test[index]))
    predict_index = np.argmax(y_hat[index])
    true_index = np.argmax(y_test[index])
    # Set the title for each image
    ax.set_title("{} ({})".format(fashion_mnist_labels[predict_index],
                                  fashion_mnist_labels[true_index]),
                                  color=("green" if predict_index == true_index else "red"))
plt.show()
