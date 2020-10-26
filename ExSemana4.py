#insira seu código aqui
import numpy as np
import cv2
import time
from tqdm import tqdm


def main():
  # Load the images
  img = imgLoad("master.jpg")
  copy = img.copy()

  # Cria a janela
  winTitle = "Ex 1 - semana 4"
  cv2.namedWindow(winTitle)

  # Cria a trackbar
  cv2.createTrackbar("Contrast", winTitle, 100, 100, onChange)
  cv2.createTrackbar("Brightness", winTitle, 0, 100, onChange)

  # Valor anterior da trackbar
  old_contrast = 100
  flag_contrast = False
  counter_time = 0

  old_brightness = 0
  flag_brightness = False


  while True:
    # Valor da trackbar
    contrast_value = cv2.getTrackbarPos("Contrast", winTitle)
    brightness_value = cv2.getTrackbarPos("Brightness", winTitle)

    # Trackbar de contraste alterada
    if old_contrast != contrast_value:
        flag_contrast = True
        counter_time = time.time()
        old_contrast = contrast_value
    # trackbar de brilho alterada
    if old_brightness != brightness_value:
        flag_brightness = True
        counter_time = time.time()
        old_brightness = brightness_value


    # Após um segundo da alteração, o contraste é atualizado
    if time.time() - counter_time > 1:
        if flag_contrast or flag_brightness:
            '''
            height, width, channels = img.shape
            copy = img.copy()

            for y in tqdm(range(height)):
                for x in range(width):
                    for c in range(channels):
                        new_color = copy[y][x][c] * (contrast_value / 100) + brightness_value
                        copy[y][x][c] = np.clip(new_color, 0, 255)
            '''
            copy = cv2.convertScaleAbs(img, alpha = contrast_value/100, beta = brightness_value)

            flag_contrast = False
            flag_brightness = False


    # Mostra a copia imagem
    cv2.imshow(winTitle, copy)

    # Se tecla 'esc' ou tecla 'q' for pressionada, fecha a aplicação
    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(1) & 0xFF == 27: break

  cv2.destroyAllWindows()

def imgLoad(name):
  image = cv2.imread(name)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  return image

def onChange(value):
    pass

main()
