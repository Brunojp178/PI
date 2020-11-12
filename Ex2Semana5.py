#insira seu código aqui
import numpy as np
import cv2


def main():
    # Load the image
    img = imgLoadGray("girafe.png")
    copy = img.copy()

    # Cria as janelas
    winTitle = "Imagem original        /       Imagem equalizada"
    cv2.namedWindow(winTitle)

    copy = cv2.equalizeHist(img)
    #stacking images side-by-side
    res = np.hstack((img,copy))

    # Show the image
    cv2.imshow(winTitle, res)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def imgLoad(name):
  image = cv2.imread(name)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  return image

def imgLoadGray(name):
  image = cv2.imread(name, 0)
  return image

main()
