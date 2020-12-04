#insira seu código aqui
import numpy as np
import cv2

def main():

    # Global variables:
    global img, winTitle

    # Load the images
    img = imgLoad("Sandman.jpg")
    copy = img.copy()
    copy = computeKmeans(copy)

    # Cria a janela
    winTitle = "Image"
    cv2.namedWindow(winTitle)
    cv2.moveWindow(winTitle, 200, 100)

    #stacking images side-by-side
    res = np.hstack((img,copy))

    cv2.imshow(winTitle, res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def imgLoad(name):
  image = cv2.imread(name)
  # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  return image

def computeKmeans(img):
    z = img.reshape((-1, 3))
    z = np.float32(z);

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.1)
    k = 16;

    _, labels, centroides = cv2.kmeans(z, k, None, criteria, 40, cv2.KMEANS_RANDOM_CENTERS)
    centroides = np.uint8(centroides)
    newImg = centroides[labels.flatten()]

    finalImg = newImg.reshape((img.shape))

    return finalImg

main()
