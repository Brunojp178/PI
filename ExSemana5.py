import numpy as np
import cv2
from matplotlib import pyplot as plt


def main():
    img = imgLoad("fall.png")
    title = "Fallguys"
    img2 = imgLoad("p&bImage.jpg")
    title2 = "P & B image"

    # Show both images
    cv2.imshow(title, img)
    cv2.imshow(title2, img2)

    color = ('r', 'g', 'b')

    # Histogram of the first image
    for i, col in enumerate(color):
        histr = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(histr, color = col)
        plt.xlim(0, 256)
    plt.show()

    # Histogram of the second image
    for i, col in enumerate(color):
        histr = cv2.calcHist([img2], [i], None, [256], [0, 256])
        plt.plot(histr, color = col)
        plt.xlim(0, 256)
    plt.show()
    
    cv2.destroyAllWindows()

def imgLoad(name):
    image = cv2.imread(name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

main()
