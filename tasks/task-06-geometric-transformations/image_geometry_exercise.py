# image_geometry_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `apply_geometric_transformations(img)` that receives a grayscale image
represented as a NumPy array (2D array) and returns a dictionary with the following transformations:

1. Translated image (shift right and down)
2. Rotated image (90 degrees clockwise)
3. Horizontally stretched image (scale width by 1.5)
4. Horizontally mirrored image (flip along vertical axis)
5. Barrel distorted image (simple distortion using a radial function)

You must use only NumPy to implement these transformations. Do NOT use OpenCV, PIL, skimage or similar libraries.

Function signature:
    def apply_geometric_transformations(img: np.ndarray) -> dict:

The return value should be like:
{
    "translated": np.ndarray,
    "rotated": np.ndarray,
    "stretched": np.ndarray,
    "mirrored": np.ndarray,
    "distorted": np.ndarray
}
"""

import numpy as np

def apply_geometric_transformations(img: np.ndarray) -> dict:

    def translate(_img : np.ndarray):
        h, w = _img.shape

        all_zeros = np.zeros(h+1,w+1, dtype=img.dtype)

        all_zeros[1:,1:] = _img
        return all_zeros
    
    def rotate(_img:np.ndarray):
        transposta = np.transpose(_img)
        # porem para que seja no sentido horário precisamos inverter em x
        transposta = transposta[:, ::-1]
        return transposta
    
    def stretche(_img : np.ndarray):
        h, w = _img.shape
        largura = int(w*1.5)

        nova = np.zeros(h,largura,dtype=_img.dtype)

        for nova_col_i in range(largura):
            col_orig = int(nova_col_i/1.5)
            col_orig = min(col_orig,w-1)

            nova[:, nova_col_i] = img[:, col_orig]


        return nova
    
    def mirror(_img : np.ndarray):
        return _img[:, ::-1]
    
    def distorted():
        height, width = img.shape
    
        # Calcula o centro da imagem
        center_y, center_x = height // 2, width // 2
        
        # Cria uma nova imagem para o resultado
        distorted = np.zeros_like(img)
        
        # Para cada pixel na imagem distorcida
        for y in range(height):
            for x in range(width):
                # Coordenadas normalizadas [-1, 1]
                norm_y = (y - center_y) / center_y
                norm_x = (x - center_x) / center_x
                
                # Distância ao quadrado do centro
                r_squared = norm_x**2 + norm_y**2
                
                # Fator de distorção (usando 1 já que não foi específicado)
                distortion = 1.0 + 1 * r_squared
                
                # Coordenadas distorcidas
                source_y = int(center_y + (y - center_y) * distortion)
                source_x = int(center_x + (x - center_x) * distortion)
                
                # Verifica se as coordenadas estão dentro dos limites
                if 0 <= source_y < height and 0 <= source_x < width:
                    distorted[y, x] = img[source_y, source_x]
        
        return distorted

    results = {}
    
    # 1. Translação
    results["translated"] = translate(img)
    
    # 2. Rotação 90 graus no sentido horário
    results["rotated"] = rotate(img)
    
    # 3. Esticamento horizontal
    results["stretched"] = stretche(img)
    
    # 4. Espelhamento horizontal
    results["mirrored"] = mirror(img)
    
    # 5. Distorção de barril
    results["distorted"] = distorted(img)
    
    return results