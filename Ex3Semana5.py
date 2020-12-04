#insira seu código aqui
# Usar a tecla 'q' para mostrar todas as imagens antes de mostrar os calculos no terminal

import numpy as np
import cv2
import math

def main():
    # Load the image
    img = imgLoadBgr("S1.jpg")
    img2 = imgLoadBgr("S2.jpg")
    img3 = imgLoadBgr("D1.jpg")
    img4 = imgLoadBgr("D2.jpg")
    img5 = imgLoadBgr("D3.jpg")

    images = [img, img2, img3, img4, img5]
    imgTitles = ["S1", "S2", "D1", "D2", "D3"]

    imagesHsv = images.copy()
    imagesGray = images.copy()

    index = 0

    # Converting images to hsv and grey
    for i in range(len(images)):
        imagesHsv[i] = cv2.cvtColor(imagesHsv[i], cv2.COLOR_RGB2HSV)
        imagesGray[i] = cv2.cvtColor(imagesGray[i], cv2.COLOR_RGB2GRAY)

    distances = imgCompare(imagesHsv)
    min = 1000000
    index_title = 0

    for i in range(1, len(distances)):
        distances[i] = int(distances[i])
        if int(distances[i]) < min:
            print(distances[i])
            min = distances[i]
            index_title = i

    print("The distance from the first image is:")
    for image in range(len(images)):
        print("Image ", image, " Distance to first = ", distances[image])

    print("\nThe image that matches the most is Image: ", imgTitles[index_title], "\nDistance: ", min)

    winTitle = "Imagem 1"
    cv2.namedWindow(winTitle)
    cv2.moveWindow(winTitle, 300, 100)

    cv2.imshow(winTitle, images[0])
    cv2.waitKey(0)

    winTitle = "Imagem 2"
    cv2.imshow(winTitle, images[1])
    cv2.waitKey(0)

    winTitle = "Imagem 3"
    cv2.imshow(winTitle, images[2])
    cv2.waitKey(0)

    winTitle = "Imagem 4"
    cv2.imshow(winTitle, images[3])
    cv2.waitKey(0)

    winTitle = "Imagem 5"
    cv2.imshow(winTitle, images[4])
    cv2.waitKey(0)

    cv2.destroyAllWindows()

# The first image on the list is the image to compare
def imgCompare(imageList):
    image_input = imageList[0]

    histograms = []
    distances = []

    h_bins = 50
    s_bins = 60
    histSize = [h_bins, s_bins]
    # hue varies from 0 to 179, saturation from 0 to 255
    h_ranges = [0, 180]
    s_ranges = [0, 256]
    # concat lists
    ranges = h_ranges + s_ranges
    # Use the 0-th and 1-st channels
    channels = [0, 1]

    histogram_input = cv2.calcHist([image_input], channels, None, histSize, ranges, accumulate=False)
    cv2.normalize(histogram_input, histogram_input, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    # For every image, calculate the histogram and save on the list
    for img in imageList:
        hist = cv2.calcHist([img], channels, None, histSize, ranges, accumulate=False)
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        histograms.append(hist)

    for histo in histograms:
        # calculate the histograms distance with the first image on the list
        correlation = cv2.compareHist(histogram_input, histo, 0)
        chiSquare = cv2.compareHist(histogram_input, histo, 1)
        bhatta = cv2.compareHist(histogram_input, histo, 3)

        correlation = pow((1/(correlation + 0.001)), 2)
        chiSquare = pow(chiSquare, 2)
        bhatta = pow(bhatta, 2)

        distance = correlation + chiSquare + bhatta
        # sqrt(Corr^2 + Chi-Sq^2 + Bhatta^2)
        distance = math.sqrt(distance)
        distances.append(distance)

    return distances

def imgLoad(name):
    image = cv2.imread(name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def imgLoadGray(name):
    image = cv2.imread(name, 0)
    return image

def imgLoadHsv(name):
    image = cv2.imread(name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return image

def imgLoadBgr(name):
    image = cv2.imread(name)
    return image

main()
