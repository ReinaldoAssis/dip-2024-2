import cv2
import numpy as np

imagem_a = 'image_a.jpg'
imagem_b = 'image_b.jpg'

f = cv2.imread(imagem_a)
g = cv2.imread(imagem_b)

if f is None or g is None:
    print("Erro ao carregar as imagens!")
    exit()

if f.shape != g.shape:
    g = cv2.resize(g, (f.shape[1], f.shape[0]))

cv2.namedWindow('Controle')

# call back vazio
def nada(x):
    pass

# cria trackbars 'a' e 'b' na mesma janela 'Controle'
cv2.createTrackbar('a', 'Controle', 0, 100, nada)
cv2.createTrackbar('b', 'Controle', 0, 100, nada)

while True:
    # lê os valores dos trackbars
    a_val = cv2.getTrackbarPos('a', 'Controle') / 100.0
    b_val = cv2.getTrackbarPos('b', 'Controle') / 100.0

    # combinação linear
    h = (a_val * f.astype(np.float32) + b_val * g.astype(np.float32)).astype(np.uint8)

    cv2.imshow('Imagem f', f)
    cv2.imshow('Imagem g', g)
    cv2.imshow('Resultado (h = a*f + b*g)', h)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
