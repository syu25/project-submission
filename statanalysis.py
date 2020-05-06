import matplotlib.pyplot as plt
import numpy as np

def plot(val_acc_hist, val_loss_hist, train_acc_hist, train_loss_hist):
    # vahist = [a.cpu().numpy() for a in val_acc_hist]
    # vlhist = [l.cpu().numpy() for l in val_loss_hist]
    # tahist = [a.cpu().numpy() for a in train_acc_hist]
    # tlhist = [l.cpu().numpy() for l in train_loss_hist]

    vahist = [a for a in val_acc_hist]
    vlhist = [l for l in val_loss_hist]
    tahist = [a for a in train_acc_hist]
    tlhist = [l for l in train_loss_hist]

    plt.figure()
    plt.title("Validation Accuracy vs. Number of Training Epochs")
    plt.xlabel("Training Epochs")
    plt.ylabel("Validation Accuracy")
    plt.plot(range(1,len(vahist)+1),vahist,label="Validation")
    plt.plot(range(1,len(tahist)+1),tahist,label="Training")
    plt.ylim((0,1.))
    plt.xticks(np.arange(1, len(vahist)+1,1.0))
    plt.legend()
    #plt.show()
    plt.figure()
    plt.title("Validation Loss vs. Number of Training Epochs")
    plt.xlabel("Training Epochs")
    plt.ylabel("Validation Loss")
    plt.plot(range(1,len(vlhist)+1), vlhist, label="Validation")
    plt.plot(range(1,len(tlhist)+1), tlhist, label="Training")
    plt.ylim((0,1.))
    plt.xticks(np.arange(1, len(vlhist)+1,1.0))
    plt.legend()
    plt.show()

