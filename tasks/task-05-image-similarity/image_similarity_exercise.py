# image_similarity_exercise.py
# STUDENT'S EXERCISE FILE
"""
Exercise:
Implement a function `compare_images(i1, i2)` that receives two grayscale images
represented as NumPy arrays (2D arrays of shape (H, W)) and returns a dictionary with the following metrics:
1. Mean Squared Error (MSE)
2. Peak Signal-to-Noise Ratio (PSNR)
3. Structural Similarity Index (SSIM) - simplified version without using external libraries
4. Normalized Pearson Correlation Coefficient (NPCC)
You must implement these functions yourself using only NumPy (no OpenCV, skimage, etc).
Each function should be implemented as a helper function and called inside `compare_images(i1, i2)`.
Function signature:
    def compare_images(i1: np.ndarray, i2: np.ndarray) -> dict:
The return value should be like:
{
    "mse": float,
    "psnr": float,
    "ssim": float,
    "npcc": float
}
Assume that i1 and i2 are normalized grayscale images (values between 0 and 1).
"""
import numpy as np

def compare_images(i1: np.ndarray, i2: np.ndarray) -> dict:
    def mse(_i1: np.ndarray, _i2: np.ndarray):
        if _i1.shape != _i2.shape:
            raise Exception("Imagem 1 e imagem 2 não possuem o mesmo tamanho!")
        erros = np.square(_i1 - _i2)
        mse_value = np.mean(erros)
        return mse_value

    def psnr(_i1: np.ndarray, _i2: np.ndarray):
        # poderiamos abstrair e não fazer essa comparação para cada um dos métodos
        # auxiliares, porém se considerarmos que esses métodos podem ser utilizados
        # por outras funções, é boa prática realizar as comparações
        if _i1.shape != _i2.shape:
            raise Exception("Imagem 1 e imagem 2 não possuem o mesmo tamanho!")
        maximo = 1
        mse_v = mse(_i1, _i2)
        if mse_v == 0:
            return float('inf')  # infinito quando as imagens são idênticas
        return 10 * np.log10(maximo**2/mse_v)

    def ssim(_i1: np.ndarray, _i2: np.ndarray):
        if _i1.shape != _i2.shape:
            raise Exception("Imagem 1 e imagem 2 não possuem o mesmo tamanho!")
        
        # definindo as constantes
        C1 = 0.01**2
        C2 = 0.03**2

        # seguindo a formula

        # media da luminosidade das imagens
        microx = np.mean(_i1)
        microy = np.mean(_i2)

        # variancia das itensidades dos pixels das imagens
        sig2x = np.var(_i1)
        sig2y = np.var(_i2)

        # covariância
        sigxy = np.mean((_i1 - microx) * (_i2 - microy))
        numerador = (2*microx*microy+C1)*(2*sigxy+C2)
        denominador = (microx**2+microy**2+C1)*(sig2x+sig2y+C2)
        return numerador/denominador

    def npcc(_i1: np.ndarray, _i2: np.ndarray):
        if _i1.shape != _i2.shape:
            raise Exception("Imagem 1 e imagem 2 não possuem o mesmo tamanho!")
        
        mean_i1 = np.mean(_i1)
        mean_i2 = np.mean(_i2)
        
        # centralização
        i1_centered = _i1 - mean_i1
        i2_centered = _i2 - mean_i2
        
        # covariância
        numerador = np.sum(i1_centered * i2_centered)
        
        std_i1 = np.sqrt(np.sum(i1_centered**2))
        std_i2 = np.sqrt(np.sum(i2_centered**2))
        denominador = std_i1 * std_i2
        
        # evitando divisão por zero
        if denominador == 0:
            return 0
            
        return numerador / denominador

    # Calculando todas as métricas e retornando o dicionário
    return {
        "mse": mse(i1, i2),
        "psnr": psnr(i1, i2),
        "ssim": ssim(i1, i2),
        "npcc": npcc(i1, i2)
    }