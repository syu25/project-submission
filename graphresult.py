from matplotlib import pyplot as plt
import pandas as pd
import os
import math


def graph_me(y_pred, y_test, title, accuracy, samples, save=False):
    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

    yticks = []

    ages = y_test.tolist()

    for x in range(int(max(ages)/4)):
        yticks.append(x*4)


    df1 = df.head(samples)

    df1.plot(kind='bar', figsize=(10,8), title=title + ": " + str(accuracy), yticks=yticks)
    plt.grid(which='major', linewidth='0.5', color='green')
    plt.grid(which='minor', linewidth='0.5', color='black')
    
    if save:
        saver = "images/" + title + ".png"
        plt.savefig(saver)
        
    plt.show()



def graph_all():
    images = []
    direc = "images/"
    for _, _, filenames in os.walk(direc):
        for string in filenames:
            images.append(direc + string)
    print(images)

    img_data = []
    for img in images:
        img_data.append(plt.imread(img))
    
    im_len = len(images)

    for i in range(im_len):
        plt.figure(i)
        plt.axis(False)
        plt.imshow(img_data[i])
        plt.tight_layout()
    plt.show()