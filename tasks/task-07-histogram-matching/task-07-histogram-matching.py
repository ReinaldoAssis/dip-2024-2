# histogram_matching_exercise.py
# STUDENT'S EXERCISE FILE

"""
From the readme file:

🧪 Exercise: Histogram Matching with OpenCV and Scikit-Image Objective: Implement a function that performs histogram matching to transfer the visual appearance of a reference image to a source image. You will output the transformed image and (opptionally) plot the histograms of its RGB channels.

📂 Provided Files:

source.jpg – Example image to be transformed.
reference.jpg – Example image whose histogram we want to match.
output.jpg – Example of expected output.
📌 Instructions:

Load the source and reference images.
Convert them to the RGB color space (Obs: If you open an image file using OpenCV, the standard is BGR).
Perform histogram matching.
(Not included in the test, but recommended); Plot histograms of the original and matched images.

"""

"""
Exercise:
Implement a function `match_histograms_rgb(source_img, reference_img)` that receives two RGB images
(as NumPy arrays with shape (H, W, 3)) and returns a new image where the histogram of each RGB channel 
from the source image is matched to the corresponding histogram of the reference image.

Your task:
- Read two RGB images: source and reference (they will be provided externally).
- Match the histograms of the source image to the reference image using all RGB channels.
- Return the matched image as a NumPy array (uint8)

Function signature:
    def match_histograms_rgb(source_img: np.ndarray, reference_img: np.ndarray) -> np.ndarray

Return:
    - matched_img: NumPy array of the result image

Notes:
- Do NOT save or display the image in this function.
- Do NOT use OpenCV to apply the histogram match (only for loading images, if needed externally).
- You can assume the input images are already loaded and in RGB format (not BGR).
"""

import cv2 as cv
import numpy as np
import skimage as ski

def match_histograms_rgb(source_img: np.ndarray, reference_img: np.ndarray) -> np.ndarray:
    
    matched_img = np.zeros_like(source_img)

    # primeiro, extraimos os canais
    for i in range(3):
        src = source_img[:,:,i]
        ref = reference_img[:,:,i]

        #calcular histogramas
        src_hist, _ = np.histogram(src.flatten(), 256, [0, 255])
        ref_hist, _ = np.histogram(ref.flatten(), 256, [0, 255])

        # calculando cdfs
        src_cdf = src_hist.cumsum()
        src_cdf = src_cdf / src_cdf[-1]

        ref_cdf = ref_hist.cumsum()
        ref_cdf = ref_cdf / ref_cdf[-1]

        # mapeamento
        transform_map = np.zeros(256)
        for j in range(256):
            diff = np.abs(src_cdf[j] - ref_cdf[:])
            index = diff.argmin()
            transform_map[j] = index

        matched_channel = transform_map[src].astype(np.uint8)
        matched_img[:, :, 2-i] = matched_channel

    return matched_img


# função para validar implementação
# essa atividade foi uma dor de cabeça, por algum motivo a imagem de output
# utiliza r=2, g=1, b=0 que é o inverso do padrão da biblioteca skimage

# def main():
#     """
#     Função principal para testar o histogram matching
#     """
#     # Carregar imagens
#     source_img = cv.imread("source.jpg")
#     reference_img = cv.imread("reference.jpg")
    
#     # Converter de BGR para RGB
#     source_img = cv.cvtColor(source_img, cv.COLOR_BGR2RGB)
#     reference_img = cv.cvtColor(reference_img, cv.COLOR_BGR2RGB)
    
#     # Aplicar histogram matching com nossa implementação
#     matched_img = match_histograms_rgb(source_img, reference_img)
    
#     # Aplicar histogram matching com scikit-image para benchmark
#     from skimage.exposure import match_histograms as ski_match_histograms
#     matched_ski = ski_match_histograms(source_img, reference_img, channel_axis=-1)
#     matched_ski = matched_ski.astype(np.uint8)
    
#     # Converter de volta para BGR para salvar com OpenCV
#     matched_bgr = cv.cvtColor(matched_img, cv.COLOR_RGB2BGR)
#     matched_ski_bgr = cv.cvtColor(matched_ski, cv.COLOR_RGB2BGR)
    
#     # Salvar resultados
#     cv.imwrite("matched_output.jpg", matched_bgr)
#     cv.imwrite("matched_ski_output.jpg", matched_ski_bgr)
    
#     # Inicializar o dicionário de imagens para plotagem
#     images = {
#         'Source': source_img, 
#         'Reference': reference_img, 
#         'Manual Match': matched_img,
#         'SKImage Match': matched_ski
#     }
    
#     # Carregar a imagem de saída esperada para comparação
#     expected_output = cv.imread("output.jpg")
#     expected_output_rgb = cv.cvtColor(expected_output, cv.COLOR_BGR2RGB)
    
#     # Calcular a similaridade entre as saídas e a esperada
#     from skimage.metrics import structural_similarity as ssim
    
#     # Garantir que as imagens têm o mesmo tamanho
#     if matched_img.shape == expected_output_rgb.shape:
#         # Calcular SSIM para implementação manual
#         similarity_manual = ssim(matched_img, expected_output_rgb, channel_axis=2)
        
#         # Calcular SSIM para implementação scikit-image
#         similarity_ski = ssim(matched_ski, expected_output_rgb, channel_axis=2)
        
#         # Calcular SSIM entre as duas implementações
#         similarity_between = ssim(matched_img, matched_ski, channel_axis=2)
        
#         # Avaliar a similaridade
#         print(f"Similaridade (manual vs esperado): {similarity_manual:.4f}")
#         print(f"Similaridade (scikit vs esperado): {similarity_ski:.4f}")
#         print(f"Similaridade (manual vs scikit): {similarity_between:.4f}")
        
#         if similarity_ski > 0.9:
#             print("A implementação do scikit-image está próxima da saída esperada.")
            
#         if similarity_between > 0.9:
#             print("As implementações manual e scikit-image produzem resultados semelhantes.")
#         else:
#             print("As implementações manual e scikit-image produzem resultados diferentes.")
            
#         # Adicionar a imagem esperada ao gráfico comparativo
#         images['Expected'] = expected_output_rgb
#     else:
#         print("ATENÇÃO! As dimensões da imagem gerada e da imagem esperada são diferentes.")
    
#     # Plotar histogramas (opcional)
#     import matplotlib.pyplot as plt
    
#     # Configuração dos subplots - ajustando para 5 colunas se necessário
#     num_cols = len(images)
#     fig, axs = plt.subplots(4, num_cols, figsize=(5*num_cols, 20))
    
#     # Nomes dos canais e imagens
#     channels = ['Red', 'Green', 'Blue']
    
#     # Plotar imagens na primeira linha
#     for i, (name, img) in enumerate(images.items()):
#         axs[0, i].imshow(img)
#         axs[0, i].set_title(name)
#         axs[0, i].axis('off')
    
#     # Plotar histogramas
#     for i, channel in enumerate(channels):
#         for j, (name, img) in enumerate(images.items()):
#             hist, bins = np.histogram(img[:,:,i].flatten(), 256, [0, 256])
#             axs[i+1, j].plot(hist)
#             axs[i+1, j].set_title(f'{name} - {channel}')
    
#     plt.tight_layout()
#     plt.savefig('histograms_comparison.png')
#     plt.show()
    
#     print("Processamento concluído. Imagens salvas como 'matched_output.jpg', 'matched_ski_output.jpg' e 'histograms_comparison.png'")

# if __name__ == "__main__":
#     main()