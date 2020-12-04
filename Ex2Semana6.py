#insira seu código aqui
import numpy as np
import cv2
from matplotlib import pyplot as plt

def main():

    # Cria a janela
    winTitle = "Imagem original"
    cv2.namedWindow(winTitle)
    cv2.moveWindow(winTitle, 200, 100)

    # Load the images
    img = imgLoad("chess.jpg")
    grabCut = img.copy()

    # cria uma copia da imagem com tudo embaçado
    embacada = applyFilter(img)
    # Aplica grabCut
    grabCut = applyGrabCut(grabCut)

    # Mostra a imagem original e espera o usuario pressionar uma tecla
    cv2.imshow(winTitle, img)
    cv2.waitKey(0)

    final = mergeImages(embacada, grabCut, 360, 505, 70, 297)

    # Mostra a imagem original e espera o usuario pressionar uma tecla
    cv2.imshow(winTitle, img)
    cv2.waitKey(0)

    # Mostra o grabCut
    winTitle = "GrabCut Image"
    cv2.imshow(winTitle, grabCut)
    cv2.waitKey(0)

    # Mostra o resultado final
    winTitle = "Result"
    cv2.imshow(winTitle, final)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

def imgLoad(name):
  image = cv2.imread(name)
  # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  return image

def imgShow(img, axis):
  imgplot = plt.imshow(img.astype('uint8'))
  if not axis: plt.axis('off')
  plt.show()

def applyGrabCut(img):

    startX = 405
    startY = 70
    x = 505
    y = 297

    #aplicar grabcut na imagem
    mask = np.zeros(img.shape[:2], np.uint8)
    rectGcut = (startX, startY, (x - startX), (y - startY))
    fundoModel = np.zeros((1, 65), np.float64)
    objModel = np.zeros((1, 65), np.float64)

    #invocar grabcut
    cv2.grabCut(img, mask, rectGcut, fundoModel, objModel, 3, cv2.GC_INIT_WITH_RECT)

    #preparando imagem final
    maskFinal = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    imgFinal = img * maskFinal[:,:,np.newaxis]
    for x in range(0, img.shape[1]):
        for y in range(0, img.shape[0]):
            if(maskFinal[y][x] == 0):
                imgFinal[y][x][0] = imgFinal[y][x][1] = imgFinal[y][x][2] = 255
    return imgFinal

def applyFilter(img):
    # Copy the original image
    img_output = img.copy()
    # todos os filtros utilizam aproximadamente valores de 0 a 10
    percentage = 7
    img_output = cv2.blur(img, (percentage, percentage))
    return img_output

def mergeImages(img_background, img_paste, x1, x2, y1, y2):

    backH, backW, _ = img_background.shape
    frontH, frontW, _ = img_paste.shape

    # recorta da imagem original a área onde a segunda imagem vai ser colocada (pasteImg)
    paste_area = img_background[y1:(y1 + frontH), x1:(x1 + frontW)]
    img_paste = img_paste[y1:(y1 + frontH), x1:(x1 + frontW)]

    # Remove fundo da imagem apagando todos os pixels brancos
    imgGrey = cv2.cvtColor(img_paste, cv2.COLOR_RGB2GRAY)
    _, maskFore = cv2.threshold(imgGrey, 250, 255, cv2.THRESH_BINARY)

    backWithMask = cv2.bitwise_and(paste_area, paste_area, mask = maskFore)
    foreWithMask = cv2.bitwise_not(maskFore)
    foreWithMask = cv2.bitwise_and(img_paste, img_paste, mask = foreWithMask)

    combinedImage = cv2.add(foreWithMask, backWithMask)

    img_background[y1:(y1 + frontH), x1:(x1+frontW)] = combinedImage

    return img_background

main()
