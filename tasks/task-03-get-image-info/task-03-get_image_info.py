import numpy as np

def get_image_info(image):
    """
    Extrai metadados e informações estatísticas de uma imagem.

    Parâmetros:
    - image (numpy.ndarray): Imagem de entrada.

    Retorna:
    - dict: Dicionário contendo metadados e estatísticas da imagem.
    """
    
    ### START CODE HERE ###

    if image.ndim == 2:
        height, width = image.shape
        depth = 1
    elif image.ndim == 3:
        height, width, depth = image.shape
    
    dtype = image.dtype
    
    min_val = np.min(image)
    max_val = np.max(image)
    mean_val = np.mean(image)
    std_val = np.std(image)
    ### END CODE HERE ###

    return {
        "width": width,
        "height": height,
        "dtype": dtype,
        "depth": depth,
        "min_value": min_val,
        "max_value": max_val,
        "mean": mean_val,
        "std_dev": std_val
    }

# Exemplo de uso:
sample_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
info = get_image_info(sample_image)

# Impressão dos resultados:
for key, value in info.items():
    print(f"{key}: {value}")
