import numpy as np
import cv2 as cv
import argparse

def load_image_from_url(url, **kwargs):
    """
    Carrega uma imagem a partir de uma URL da Internet, permitindo argumentos opcionais para cv.imdecode.

    Parâmetros:
    - url (str): URL da imagem.
    - **kwargs: Argumentos opcionais para cv.imdecode (por exemplo, flags=cv.IMREAD_GRAYSCALE).

    Retorna:
    - image: Imagem carregada como um array NumPy.
    """

    ### START CODE HERE ###
    import urllib.request # IMPORTANDO AQUI POIS NÃO SEI SE PODEMOS ALTERAR FORA DESSA REGIÃO
    with urllib.request.urlopen(url) as response:
        image_data = response.read()
    image_array = np.asarray(bytearray(image_data), dtype=np.uint8)
    flags = kwargs.get('flags', cv.IMREAD_COLOR)
    image = cv.imdecode(image_array, flags)
    ### END CODE HERE ###
    
    return image

load_image_from_url()