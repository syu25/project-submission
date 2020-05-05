import os


def delete_contents(x):
    if x == 0:
        file_path = "machines/"
    if x == 1:
        file_path = "images/"

    for i in os.listdir(file_path):
        os.remove(os.path.join(file_path, i))
