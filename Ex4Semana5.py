#insira seu código aqui
import numpy as np
import cv2
from cv2 import VideoWriter, VideoWriter_fourcc
import time
from tqdm import tqdm

# Crie duas funções chamada videoErosao e videoDilatacao.
# A função videoErosao() receberá uma imagem em preto e branco como a apresentada abaixo e deverá aplicar a operação de erosão sucessivamente, até ela desaparecer. Faça um vídeo que aplique essa operação aos poucos, em formato de animação.
# Realize procedimento parecido para videoDilatacao(), só que fazendo o oposto com a operação de dilatação.

def main():

    # Global variables:
    global img, winTitle, winTitle2, video, video2

    # Load the images
    img = imgLoad("b.jpg")

    # Video variables based on image
    height, width, _ = img.shape
    FPS = 30

    # Video encoder and name
    fourcc = VideoWriter_fourcc(*'MP42')
    video = VideoWriter('./erosion.avi', fourcc, float(FPS), (width, height))
    video2 = VideoWriter('./dilation.avi', fourcc, float(FPS), (width, height))

    # Cria a janela
    winTitle = "Erosion"
    cv2.namedWindow(winTitle)
    cv2.moveWindow(winTitle, 200, 100)

    # Cria a janela
    winTitle2 = "Dilation"
    cv2.namedWindow(winTitle2)
    cv2.moveWindow(winTitle2, 200 + width, 100)

    # Cria as trackbars
    cv2.createTrackbar("Erode", winTitle, 0, 50, erode)
    cv2.createTrackbar("Dilation", winTitle2, 0, 250, dilatation)

    while True:

        erode(0)
        dilatation(0)

        # Se tecla 'esc' ou tecla 'q' for pressionada, fecha a aplicação
        if cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(1) & 0xFF == 27: break

    video.release()
    video2.release()
    cv2.destroyAllWindows()

def imgLoad(name):
  image = cv2.imread(name)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  return image

def erode(value):
    # Erosion variables
    erosion_size = 0
    max_elem = 2
    max_kernel_size = 21

    erosion_size = cv2.getTrackbarPos("Erode", winTitle)

    # Erosion types: cv2.MORPH_RECT; cv2.MORPH_CROSS; cv.MORPH_ELLIPSE
    erosion_type = cv2.MORPH_RECT

    element = cv2.getStructuringElement(erosion_type, (2*erosion_size + 1, 2*erosion_size+1), (erosion_size, erosion_size))
    erosion_dst = cv2.erode(img, element)
    if erosion_size != 0:   video.write(erosion_dst)
    cv2.imshow(winTitle, erosion_dst)

def dilatation(val):
    # Dilation variables
    dilatation_size = cv2.getTrackbarPos("Dilation", winTitle2)

    # Dilation types: cv2.MORPH_RECT; cv2.MORPH_CROSS; cv2.MORPH_ELLIPSE
    dilatation_type = cv2.MORPH_RECT

    element = cv2.getStructuringElement(dilatation_type, (2*dilatation_size + 1, 2*dilatation_size+1), (dilatation_size, dilatation_size))
    dilatation_dst = cv2.dilate(img, element)
    if dilatation_size != 0:    video2.write(dilatation_dst)
    cv2.imshow(winTitle2, dilatation_dst)

main()
