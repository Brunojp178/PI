import numpy as np
import cv2
from matplotlib import pyplot as plt


def main():
    img = imgLoad("master.jpg")

    # Filters -------------------------------------------
    # Value between [0, 100]
    filter_percentage = 200

    filter_percentage = np.clip(filter_percentage, 0, 99)

    # media
    media = applyFilter(img, "media", filter_percentage)
    # Gauss filter
    gauss = applyFilter(img, "Gauss", filter_percentage)
    # Median filter
    mediana = applyFilter(img, "mediana", filter_percentage)
    # Sobel Horizontal
    sobelX = applyFilter(img, "sobelHorizontal", filter_percentage)
    # Sobel Vertical
    sobelY = applyFilter(img, "sobelVertical", filter_percentage)
    # Laplacian
    laplacian = applyFilter(img, "laplaciano", filter_percentage)


    imageList = [media, gauss, mediana, sobelX, sobelY, laplacian]
    imageTitles = ["Média", "Gauss", "Mediana", "Sobel H.", "Sobel V.", "Laplaciano"]

    # col, lin

    colunas = 2
    linhas = 3
    showImageGrid(imageList, imageTitles, colunas, linhas)

    cv2.waitKey(0)

    cv2.destroyAllWindows()

# Função que aplica os filtros
def applyFilter(img, filter_name, percentage):

    # possible filters
    filter_name = filter_name.lower()
    filters = ["media", "gauss", "mediana", "sobelvertical", "sobelhorizontal", "laplaciano"]
    if not (filter_name in filters): print("Filtro não encontrado!"); return
    # Copy the original image
    img_output = img.copy()
    # todos os filtros utilizam aproximadamente valores de 0 a 10
    percentage = int(percentage/10)
    if percentage < 1: percentage = 1; print("Argumento menor que 1, resetado!")


    # filtro de média
    if filter_name == "media":
        img_output = cv2.blur(img, (percentage, percentage))
    # Filtro de Gauss
    if filter_name == "gauss":
        if percentage % 2 == 0:
            # Se a var percentage for um número par, subtrai um e divide por 10 (60 -> 5,9 -> 5)
            # e se essa conta for menor que 1, reseta o valor para 1
            percentage -= 1
            img_output = cv2.GaussianBlur(img, (percentage, percentage), 0)
        else:
            img_output = cv2.GaussianBlur(img, (percentage, percentage), 0)

    # Filtro de mediana!
    if filter_name == "mediana":
        img_output = cv2.medianBlur(img, percentage)

    # Filtro Sobel
    if filter_name == "sobelvertical":
        img_output = cv2.Sobel(img, cv2.CV_64F, 0, 1, percentage)
        img_output = np.uint8(np.absolute(img_output))

    if filter_name == "sobelhorizontal":
        img_output = cv2.Sobel(img, cv2.CV_64F, 1, 0, percentage)
        img_output = np.uint8(np.absolute(img_output))

    if filter_name == "laplaciano":
        img_output = cv2.Laplacian(img, cv2.CV_64F, percentage)
        img_output = np.uint8(np.absolute(img_output))

    return img_output

def showImageGrid(imgList, titleList, x, y):
    if x < 1 or y < 1: print("ERRO: Valores de X e Y não podem ser menores que 1"); return

    fig, axis = plt.subplots(y, x)
    xIndex, yIndex, titleIndex = 0, 0, 0

    for img in imgList:
        axis[yIndex, xIndex].set_title(titleList[titleIndex])
        axis[yIndex, xIndex].imshow(img)

        axis[yIndex,xIndex].axis('off')

        titleIndex += 1
        xIndex += 1

        if xIndex == x:
            xIndex = 0
            yIndex += 1

    fig.tight_layout(pad=0.2)
    plt.show()

def imgLoad(name):
    image = cv2.imread(name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

main()
