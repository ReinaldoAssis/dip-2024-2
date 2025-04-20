""""
1. Display Color Histograms for RGB Images
Objective: Calculate and display separate histograms for the R, G, and B channels of a color image.
Topics: Color histograms, channel separation.
Challenge: Compare histograms of different images (e.g., nature vs. synthetic images).

---

2. Visualize Individual Color Channels
Objective: Extract and display the Red, Green, and Blue channels of a color image as grayscale and pseudo-colored images.
Topics: Channel separation and visualization.
Bonus: Reconstruct the original image using the separated channels.

---

3. Convert Between Color Spaces (RGB ↔ HSV, LAB, YCrCb, CMYK)
Objective: Convert an RGB image to other color spaces and display the result.
Topics: Color space conversion.
Challenge: Display individual channels from each converted space.

---

4. Compare Effects of Blurring in RGB vs HSV
Objective: Apply Gaussian blur in both RGB and HSV color spaces and compare results.
Topics: Color space effect on filtering.
Discussion: Why HSV might preserve color better in some operations.

---
5. Apply Edge Detection Filters (Sobel, Laplacian) on Color Images
Objective: Apply Sobel and Laplacian filters on individual channels and on the grayscale version of the image.
Topics: Edge detection, spatial filtering.
Bonus: Merge edge maps from all channels to form a combined result.

---

6. High-pass and Low-pass Filtering in the Frequency Domain
Objective: Perform DFT on each channel of a color image, apply high-pass and low-pass masks, then reconstruct the image.
Topics: Frequency domain filtering, Fourier Transform.
Tools: cv2.dft(), cv2.idft(), numpy.fft.
---

7. Visualize and Manipulate Bit Planes of Color Images
Objective: Extract bit planes (especially MSB and LSB) from each color channel and display them.
Topics: Bit slicing, data visualization.
Challenge: Reconstruct the image using only the top 4 bits of each channel.

---

8. Color-based Object Segmentation using HSV Thresholding
Objective: Convert image to HSV, apply thresholding to extract objects of a certain color (e.g., red apples).
Topics: Color segmentation, HSV masking.
Bonus: Use trackbars to adjust thresholds dynamically.

---

9. Convert and Visualize Images in the NTSC (YIQ) Color Space
Objective: Manually convert an RGB image to NTSC (YIQ) and visualize the Y, I, and Q channels.
Topics: Color space math, visualization.
Note: OpenCV doesn’t support YIQ directly, so students can implement the conversion using matrices.

---

10. Color Image Enhancement with Histogram Equalization
Objective: Apply histogram equalization on individual channels in different color spaces (e.g., Y in YCrCb, L in LAB).
Topics: Contrast enhancement, color models.
Discussion: Explain why histogram equalization should not be directly applied to RGB.

"""



import sys
import numpy as np
from skimage import io, color, exposure
import traceback
import matplotlib.pyplot as plt
from scipy import ndimage
import cv2

def load_image():
    """Load and return the image for processing."""
    return io.imread('../src/pdi/rgb.png')

def task1():
    """Display Color Histograms for RGB Images."""
    print("Executando Tarefa 1: Histogramas de cores para imagens RGB")
    
    image = load_image()
    red_channel = image[:, :, 0]
    green_channel = image[:, :, 1]
    blue_channel = image[:, :, 2]

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.hist(red_channel.ravel(), bins=256, color='red', alpha=0.7)
    plt.title('Vermelho')

    plt.subplot(1, 3, 2)
    plt.hist(green_channel.ravel(), bins=256, color='green', alpha=0.7)
    plt.title('Verde')

    plt.subplot(1, 3, 3)
    plt.hist(blue_channel.ravel(), bins=256, color='blue', alpha=0.7)
    plt.title('Azul')

    plt.tight_layout()
    plt.show()

def task2():
    """Visualize Individual Color Channels."""
    print("Executando Tarefa 2: Visualização dos canais de cores individuais")
    
    image = load_image()
    red_channel = image[:, :, 0]
    green_channel = image[:, :, 1]
    blue_channel = image[:, :, 2]

   
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(red_channel, cmap='gray')
    plt.title('Canal Vermelho (Escala de Cinza)')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(green_channel, cmap='gray')
    plt.title('Canal Verde (Escala de Cinza)')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(blue_channel, cmap='gray')
    plt.title('Canal Azul (Escala de Cinza)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def task3():
    """Convert Between Color Spaces (RGB ↔ HSV, LAB, YCrCb, CMYK)"""
    print("Executando Tarefa 3: Conversão entre espaços de cores")
    
   
    image = load_image()
    
   
    hsv_image = color.rgb2hsv(image)
    lab_image = color.rgb2lab(image)
    ycrcb_image = color.rgb2ycbcr(image)
    
   
   
    r, g, b = image[:,:,0]/255.0, image[:,:,1]/255.0, image[:,:,2]/255.0
    k = np.minimum(1-r, np.minimum(1-g, 1-b))
    c = (1-r-k)/(1-k+1e-10) 
    m = (1-g-k)/(1-k+1e-10)
    y = (1-b-k)/(1-k+1e-10)
    cmyk_image = np.dstack((c, m, y, k))
    
   
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.imshow(image)
    plt.title('RGB Original')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(hsv_image)
    plt.title('HSV')
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(lab_image)
    plt.title('LAB')
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.imshow(ycrcb_image.astype('uint8'))
    plt.title('YCrCb')
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.imshow(np.dstack((c, m, y)))
    plt.title('CMY')
    plt.axis('off')
    
    plt.subplot(2, 3, 6)
    plt.imshow(k, cmap='gray')
    plt.title('K (Preto)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
   
   
    plt.figure(figsize=(15, 15))
    
    plt.subplot(4, 3, 1)
    plt.imshow(hsv_image[:,:,0], cmap='hsv')
    plt.title('Matiz (H)')
    plt.axis('off')
    
    plt.subplot(4, 3, 2)
    plt.imshow(hsv_image[:,:,1], cmap='gray')
    plt.title('Saturação (S)')
    plt.axis('off')
    
    plt.subplot(4, 3, 3)
    plt.imshow(hsv_image[:,:,2], cmap='gray')
    plt.title('Valor (V)')
    plt.axis('off')
    
   
    plt.subplot(4, 3, 4)
    plt.imshow(lab_image[:,:,0], cmap='gray')
    plt.title('Luminosidade (L)')
    plt.axis('off')
    
    plt.subplot(4, 3, 5)
    plt.imshow(lab_image[:,:,1], cmap='RdYlGn')
    plt.title('Canal a (verde-vermelho)')
    plt.axis('off')
    
    plt.subplot(4, 3, 6)
    plt.imshow(lab_image[:,:,2], cmap='coolwarm')
    plt.title('Canal b (azul-amarelo)')
    plt.axis('off')
    
   
    plt.subplot(4, 3, 7)
    plt.imshow(ycrcb_image[:,:,0], cmap='gray')
    plt.title('Luminância (Y)')
    plt.axis('off')
    
    plt.subplot(4, 3, 8)
    plt.imshow(ycrcb_image[:,:,1], cmap='Reds')
    plt.title('Crominância Vermelha (Cr)')
    plt.axis('off')
    
    plt.subplot(4, 3, 9)
    plt.imshow(ycrcb_image[:,:,2], cmap='Blues')
    plt.title('Crominância Azul (Cb)')
    plt.axis('off')
    
   
    plt.subplot(4, 3, 10)
    plt.imshow(c, cmap='Cyan')
    plt.title('Ciano (C)')
    plt.axis('off')
    
    plt.subplot(4, 3, 11)
    plt.imshow(m, cmap='magenta')
    plt.title('Magenta (M)')
    plt.axis('off')
    
    plt.subplot(4, 3, 12)
    plt.imshow(y, cmap='yellow')
    plt.title('Amarelo (Y)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def task4():
    """Compare Effects of Blurring in RGB vs HSV"""
    print("Executando Tarefa 4: Comparação dos efeitos de desfoque em RGB vs HSV")
    
   
    image = load_image()
    
   
    kernel_size = 15
    sigma = 5
    
   
    blurred_rgb = ndimage.gaussian_filter(image, sigma=(sigma, sigma, 0))
    
   
    hsv_image = color.rgb2hsv(image)
   
    hsv_image[:,:,2] = ndimage.gaussian_filter(hsv_image[:,:,2], sigma=sigma)
    blurred_hsv_rgb = color.hsv2rgb(hsv_image)
    
   
    blurred_hsv_rgb = np.clip(blurred_hsv_rgb * 255, 0, 255).astype(np.uint8)
    
   
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title('Imagem Original')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(blurred_rgb)
    plt.title(f'Desfoque Gaussiano em RGB (sigma={sigma})')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(blurred_hsv_rgb)
    plt.title(f'Desfoque Gaussiano em HSV (apenas V, sigma={sigma})')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
   
    print("\nDISCUSSÃO: Por que o HSV preserva melhor as cores em algumas operações?")
    print("1. No espaço HSV, a informação de cor (matiz e saturação) é separada da informação de brilho (valor).")
    print("2. Ao aplicar o desfoque apenas no canal de valor (V), as cores originais são preservadas.")
    print("3. No espaço RGB, qualquer operação afeta simultaneamente as três dimensões de cor.")
    print("4. O HSV é mais intuitivo para modificações seletivas: alterar o brilho sem afetar as cores.")
    print("5. Em operações como desfoque, suavização ou equalização, o HSV permite preservar características perceptíveis da imagem.")

def task5():
    """Apply Edge Detection Filters (Sobel, Laplacian) on Color Images"""
    print("Executando Tarefa 5: Aplicação de filtros de detecção de bordas")
    
   
    image = load_image()
    
   
    gray_image = color.rgb2gray(image)
    
   
    sobel_gray_h = ndimage.sobel(gray_image, axis=0)
    sobel_gray_v = ndimage.sobel(gray_image, axis=1)
    sobel_gray_magnitude = np.sqrt(sobel_gray_h**2 + sobel_gray_v**2)
    sobel_gray_magnitude = exposure.rescale_intensity(sobel_gray_magnitude, out_range=(0, 1))
    
    laplacian_gray = ndimage.laplace(gray_image)
    laplacian_gray = exposure.rescale_intensity(np.abs(laplacian_gray), out_range=(0, 1))
    
   
    sobel_red_h = ndimage.sobel(image[:,:,0], axis=0)
    sobel_red_v = ndimage.sobel(image[:,:,0], axis=1)
    sobel_red = np.sqrt(sobel_red_h**2 + sobel_red_v**2)
    sobel_red = exposure.rescale_intensity(sobel_red, out_range=(0, 1))
    
    sobel_green_h = ndimage.sobel(image[:,:,1], axis=0)
    sobel_green_v = ndimage.sobel(image[:,:,1], axis=1)
    sobel_green = np.sqrt(sobel_green_h**2 + sobel_green_v**2)
    sobel_green = exposure.rescale_intensity(sobel_green, out_range=(0, 1))
    
    sobel_blue_h = ndimage.sobel(image[:,:,2], axis=0)
    sobel_blue_v = ndimage.sobel(image[:,:,2], axis=1)
    sobel_blue = np.sqrt(sobel_blue_h**2 + sobel_blue_v**2)
    sobel_blue = exposure.rescale_intensity(sobel_blue, out_range=(0, 1))
    
    laplacian_red = np.abs(ndimage.laplace(image[:,:,0]))
    laplacian_red = exposure.rescale_intensity(laplacian_red, out_range=(0, 1))
    
    laplacian_green = np.abs(ndimage.laplace(image[:,:,1]))
    laplacian_green = exposure.rescale_intensity(laplacian_green, out_range=(0, 1))
    
    laplacian_blue = np.abs(ndimage.laplace(image[:,:,2]))
    laplacian_blue = exposure.rescale_intensity(laplacian_blue, out_range=(0, 1))
    
   
    sobel_combined = np.maximum.reduce([sobel_red, sobel_green, sobel_blue])
    laplacian_combined = np.maximum.reduce([laplacian_red, laplacian_green, laplacian_blue])
    
   
   
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.imshow(image)
    plt.title('Imagem Original')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(sobel_gray_magnitude, cmap='gray')
    plt.title('Sobel - Escala de Cinza')
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(sobel_combined, cmap='gray')
    plt.title('Sobel - Combinado RGB')
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.imshow(sobel_red, cmap='gray')
    plt.title('Sobel - Canal Vermelho')
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.imshow(sobel_green, cmap='gray')
    plt.title('Sobel - Canal Verde')
    plt.axis('off')
    
    plt.subplot(2, 3, 6)
    plt.imshow(sobel_blue, cmap='gray')
    plt.title('Sobel - Canal Azul')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

   
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.imshow(image)
    plt.title('Imagem Original')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(laplacian_gray, cmap='gray')
    plt.title('Laplaciano - Escala de Cinza')
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(laplacian_combined, cmap='gray')
    plt.title('Laplaciano - Combinado RGB')
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.imshow(laplacian_red, cmap='gray')
    plt.title('Laplaciano - Canal Vermelho')
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.imshow(laplacian_green, cmap='gray')
    plt.title('Laplaciano - Canal Verde')
    plt.axis('off')
    
    plt.subplot(2, 3, 6)
    plt.imshow(laplacian_blue, cmap='gray')
    plt.title('Laplaciano - Canal Azul')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
   
    colored_edges = np.zeros_like(image)
    colored_edges[:,:,0] = sobel_red 
    colored_edges[:,:,1] = sobel_green 
    colored_edges[:,:,2] = sobel_blue 
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Imagem Original')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(colored_edges)
    plt.title('Mapa de Bordas Colorido (Sobel)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def task6():
    """High-pass and Low-pass Filtering in the Frequency Domain"""
    print("Executando Tarefa 6: Filtragem passa-alta e passa-baixa no domínio da frequência")
    
   
    image = load_image()
    
   
    channels = cv2.split(image)
    filtered_channels_lowpass = []
    filtered_channels_highpass = []
    dft_channels = []
    
    for i, channel in enumerate(channels):
       
        rows, cols = channel.shape
        padded = cv2.copyMakeBorder(channel, 0, rows, 0, cols, cv2.BORDER_CONSTANT, value=0)
        
       
        dft = cv2.dft(np.float32(padded), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        
       
        magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1)
        dft_channels.append(magnitude_spectrum)
        
       
        center_row, center_col = rows, cols
        radius_low = min(center_row, center_col) // 8 
        
        mask_low = np.zeros((rows * 2, cols * 2, 2), np.float32)
        mask_high = np.ones((rows * 2, cols * 2, 2), np.float32)
        
        for y in range(rows * 2):
            for x in range(cols * 2):
                distance = np.sqrt((y - center_row) ** 2 + (x - center_col) ** 2)
               
                if distance < radius_low:
                    mask_low[y, x] = 1
                    mask_high[y, x] = 0 
        
       
        fshift_low = dft_shift * mask_low
        fshift_high = dft_shift * mask_high
        
       
        f_ishift_low = np.fft.ifftshift(fshift_low)
        img_back_low = cv2.idft(f_ishift_low)
        img_back_low = cv2.magnitude(img_back_low[:, :, 0], img_back_low[:, :, 1])
       
        filtered_low = cv2.normalize(img_back_low, None, 0, 255, cv2.NORM_MINMAX)
        filtered_channels_lowpass.append(filtered_low[:rows, :cols].astype(np.uint8))
        
        f_ishift_high = np.fft.ifftshift(fshift_high)
        img_back_high = cv2.idft(f_ishift_high)
        img_back_high = cv2.magnitude(img_back_high[:, :, 0], img_back_high[:, :, 1])
       
        filtered_high = cv2.normalize(img_back_high, None, 0, 255, cv2.NORM_MINMAX)
        filtered_channels_highpass.append(filtered_high[:rows, :cols].astype(np.uint8))
    
   
    lowpass_image = cv2.merge(filtered_channels_lowpass)
    highpass_image = cv2.merge(filtered_channels_highpass)
    
   
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.imshow(image)
    plt.title('Imagem Original')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(lowpass_image)
    plt.title('Filtro Passa-Baixa')
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(highpass_image)
    plt.title('Filtro Passa-Alta')
    plt.axis('off')
    
   
    plt.subplot(2, 3, 4)
    plt.imshow(dft_channels[0], cmap='gray')
    plt.title('Espectro - Canal Vermelho')
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.imshow(dft_channels[1], cmap='gray')
    plt.title('Espectro - Canal Verde')
    plt.axis('off')
    
    plt.subplot(2, 3, 6)
    plt.imshow(dft_channels[2], cmap='gray')
    plt.title('Espectro - Canal Azul')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def task7():
    """Visualize and Manipulate Bit Planes of Color Images"""
    print("Executando Tarefa 7: Visualização e manipulação de planos de bits")
    
   
    image = load_image()
    
   
    channels = cv2.split(image)
    channel_names = ['Vermelho', 'Verde', 'Azul']
    
   
    for c, (channel, name) in enumerate(zip(channels, channel_names)):
        plt.figure(figsize=(15, 8))
        plt.suptitle(f'Planos de Bits - Canal {name}', fontsize=16)
        
       
        for bit in range(8):
            plt.subplot(2, 4, bit+1)
           
            bit_plane = np.right_shift(channel, bit) & 1
            bit_plane = bit_plane * 255 
            plt.imshow(bit_plane, cmap='gray')
            if bit == 0:
                plt.title(f'Bit {bit} (LSB)')
            elif bit == 7:
                plt.title(f'Bit {bit} (MSB)')
            else:
                plt.title(f'Bit {bit}')
            plt.axis('off')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()
    
   
    reconstructed_channels = []
    
    plt.figure(figsize=(15, 5))
    
    for c, (channel, name) in enumerate(zip(channels, channel_names)):
       
        msb_only = channel & 0xF0 
        reconstructed_channels.append(msb_only)
        
       
        plt.subplot(1, 3, c+1)
        plt.imshow(msb_only, cmap='gray')
        plt.title(f'Canal {name} - 4 MSBs')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
   
    reconstructed_image = cv2.merge(reconstructed_channels)
    
   
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Imagem Original')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_image)
    plt.title('Imagem Reconstruída (4 MSBs)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
   
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Efeito da Remoção Progressiva de Bits (LSB primeiro)', fontsize=16)
    axes = axes.flatten()
    
    for i in range(8):
       
        mask = 0xFF << i 
        filtered_channels = [channel & mask for channel in channels]
        filtered_image = cv2.merge(filtered_channels)
        
        axes[i].imshow(filtered_image)
        axes[i].set_title(f'Usando bits {i}-7')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

def task8():
    """Color-based Object Segmentation using HSV Thresholding"""
    print("Executando Tarefa 8: Segmentação de objetos baseada em cores usando limiarização HSV")
    
   
    image = load_image()
    
   
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
   
    lower_hsv = np.array([0, 100, 100]) 
    upper_hsv = np.array([10, 255, 255]) 
    
   
    mask = cv2.inRange(hsv_image, lower_hsv, upper_hsv)
    
   
    segmented = cv2.bitwise_and(image, image, mask=mask)
    
   
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title('Imagem Original')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap='gray')
    plt.title('Máscara HSV')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(segmented)
    plt.title('Objetos Segmentados')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
   
    try:
       
        window_name = 'HSV Thresholding'
        cv2.namedWindow(window_name)
        
       
        def on_trackbar_change(val):
           
            h_min = cv2.getTrackbarPos('H Min', window_name)
            h_max = cv2.getTrackbarPos('H Max', window_name)
            s_min = cv2.getTrackbarPos('S Min', window_name)
            s_max = cv2.getTrackbarPos('S Max', window_name)
            v_min = cv2.getTrackbarPos('V Min', window_name)
            v_max = cv2.getTrackbarPos('V Max', window_name)
            
           
            lower_hsv = np.array([h_min, s_min, v_min])
            upper_hsv = np.array([h_max, s_max, v_max])
            
           
            mask = cv2.inRange(hsv_image, lower_hsv, upper_hsv)
            result = cv2.bitwise_and(image, image, mask=mask)
            
           
            cv2_image = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            cv2.imshow(window_name, cv2_image)
        
       
        cv2.createTrackbar('H Min', window_name, 0, 179, on_trackbar_change)
        cv2.createTrackbar('H Max', window_name, 10, 179, on_trackbar_change)
        cv2.createTrackbar('S Min', window_name, 100, 255, on_trackbar_change)
        cv2.createTrackbar('S Max', window_name, 255, 255, on_trackbar_change)
        cv2.createTrackbar('V Min', window_name, 100, 255, on_trackbar_change)
        cv2.createTrackbar('V Max', window_name, 255, 255, on_trackbar_change)
        
       
        on_trackbar_change(0)
        
        print("\nBonus: Interface interativa criada com trackbars.")
        print("Ajuste os controles deslizantes para segmentar diferentes cores.")
        print("Pressione ESC para sair da interface interativa.")
        
       
        while True:
            k = cv2.waitKey(1) & 0xFF
            if k == 27: 
                break
        
        cv2.destroyAllWindows()
    
    except Exception as e:
        print(f"\nAviso: A funcionalidade de trackbars interativos não está disponível neste ambiente.")
        print(f"Erro: {e}")
        
       
        plt.figure(figsize=(15, 10))
        
       
        lower_red = np.array([0, 100, 100])
        upper_red = np.array([10, 255, 255])
        mask_red = cv2.inRange(hsv_image, lower_red, upper_red)
        segmented_red = cv2.bitwise_and(image, image, mask=mask_red)
        
        plt.subplot(2, 3, 1)
        plt.imshow(segmented_red)
        plt.title('Segmentação - Vermelho')
        plt.axis('off')
        
       
        lower_green = np.array([40, 100, 100])
        upper_green = np.array([80, 255, 255])
        mask_green = cv2.inRange(hsv_image, lower_green, upper_green)
        segmented_green = cv2.bitwise_and(image, image, mask=mask_green)
        
        plt.subplot(2, 3, 2)
        plt.imshow(segmented_green)
        plt.title('Segmentação - Verde')
        plt.axis('off')
        
       
        lower_blue = np.array([100, 100, 100])
        upper_blue = np.array([140, 255, 255])
        mask_blue = cv2.inRange(hsv_image, lower_blue, upper_blue)
        segmented_blue = cv2.bitwise_and(image, image, mask=mask_blue)
        
        plt.subplot(2, 3, 3)
        plt.imshow(segmented_blue)
        plt.title('Segmentação - Azul')
        plt.axis('off')
        
       
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([35, 255, 255])
        mask_yellow = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
        segmented_yellow = cv2.bitwise_and(image, image, mask=mask_yellow)
        
        plt.subplot(2, 3, 4)
        plt.imshow(segmented_yellow)
        plt.title('Segmentação - Amarelo')
        plt.axis('off')
        
       
        lower_cyan = np.array([80, 100, 100])
        upper_cyan = np.array([100, 255, 255])
        mask_cyan = cv2.inRange(hsv_image, lower_cyan, upper_cyan)
        segmented_cyan = cv2.bitwise_and(image, image, mask=mask_cyan)
        
        plt.subplot(2, 3, 5)
        plt.imshow(segmented_cyan)
        plt.title('Segmentação - Ciano')
        plt.axis('off')
        
       
        lower_magenta = np.array([140, 100, 100])
        upper_magenta = np.array([160, 255, 255])
        mask_magenta = cv2.inRange(hsv_image, lower_magenta, upper_magenta)
        segmented_magenta = cv2.bitwise_and(image, image, mask=mask_magenta)
        
        plt.subplot(2, 3, 6)
        plt.imshow(segmented_magenta)
        plt.title('Segmentação - Magenta')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

def task9():
    """Convert and Visualize Images in the NTSC (YIQ) Color Space"""
    print("Executando Tarefa 9: Conversão e visualização de imagens no espaço de cores NTSC (YIQ)")
    
   
    image = load_image()
    
   
    image_normalized = image / 255.0
    
   
    rgb_to_yiq_matrix = np.array([
        [0.299, 0.587, 0.114],
        [0.596, -0.274, -0.322],
        [0.211, -0.523, 0.312]
    ])
    
   
    yiq_image = np.zeros_like(image_normalized)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            yiq_image[i, j] = np.dot(rgb_to_yiq_matrix, image_normalized[i, j])
    
   
    y_channel = yiq_image[:, :, 0]
    i_channel = yiq_image[:, :, 1]
    q_channel = yiq_image[:, :, 2]
    
   
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.imshow(image)
    plt.title('Imagem RGB Original')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(y_channel, cmap='gray')
    plt.title('Y - Luminância')
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.imshow(i_channel, cmap='RdBu')
    plt.title('I - Crominância laranja-azul')
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    plt.imshow(q_channel, cmap='PiYG')
    plt.title('Q - Crominância verde-magenta')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
   
   
    yiq_to_rgb_matrix = np.linalg.inv(rgb_to_yiq_matrix)
    
   
    reconstructed_image = np.zeros_like(yiq_image)
    for i in range(yiq_image.shape[0]):
        for j in range(yiq_image.shape[1]):
            reconstructed_image[i, j] = np.dot(yiq_to_rgb_matrix, yiq_image[i, j])
    
   
    reconstructed_image = np.clip(reconstructed_image, 0, 1)
    
   
    reconstructed_image = (reconstructed_image * 255).astype(np.uint8)
    
   
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Imagem Original')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_image)
    plt.title('Imagem Reconstruída (YIQ → RGB)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
   
    print("\nSobre o espaço de cores YIQ:")
    print("1. Foi desenvolvido para o sistema de TV analógica NTSC nos EUA.")
    print("2. Y representa luminância (brilho), similar ao Y em YCrCb.")
    print("3. I e Q representam componentes de crominância (cor).")
    print("4. Y varia de 0 a 1, I varia de -0.6 a +0.6, e Q varia de -0.5 a +0.5.")
    print("5. É útil para compressão de imagem e separação de informação de luminância/crominância.")
    print("6. Hoje em dia, é menos comum que outros espaços de cores como HSV e YCrCb.")

def task10():
    """Color Image Enhancement with Histogram Equalization"""
    print("Executando Tarefa 10: Aprimoramento de imagem colorida com equalização de histograma")
    
   
    image = load_image()
    
   
    equalized_rgb = np.zeros_like(image)
    for i in range(3):
        equalized_rgb[:, :, i] = exposure.equalize_hist(image[:, :, i]) * 255
    
   
    ycrcb_image = color.rgb2ycbcr(image)
    ycrcb_equalized = ycrcb_image.copy()
    ycrcb_equalized[:, :, 0] = exposure.equalize_hist(ycrcb_image[:, :, 0]) * 255
    rgb_from_ycrcb_equalized = color.ycbcr2rgb(ycrcb_equalized)
    rgb_from_ycrcb_equalized = np.clip(rgb_from_ycrcb_equalized, 0, 1) * 255
    
   
    lab_image = color.rgb2lab(image)
    lab_equalized = lab_image.copy()
    
   
    L = lab_image[:, :, 0]
    L_min, L_max = L.min(), L.max()
    L_norm = (L - L_min) / (L_max - L_min)
    
   
    lab_equalized[:, :, 0] = exposure.equalize_hist(L_norm) * (L_max - L_min) + L_min
    
    rgb_from_lab_equalized = color.lab2rgb(lab_equalized)
    rgb_from_lab_equalized = np.clip(rgb_from_lab_equalized, 0, 1) * 255
    
   
    ycrcb_clahe = ycrcb_image.copy()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    ycrcb_clahe[:, :, 0] = clahe.apply(ycrcb_image[:, :, 0].astype(np.uint8))
    rgb_from_ycrcb_clahe = color.ycbcr2rgb(ycrcb_clahe)
    rgb_from_ycrcb_clahe = np.clip(rgb_from_ycrcb_clahe, 0, 1) * 255
    
   
    plt.figure(figsize=(16, 12))
    
    plt.subplot(2, 3, 1)
    plt.imshow(image)
    plt.title('Imagem Original')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(equalized_rgb.astype(np.uint8))
    plt.title('Equalização Direta RGB')
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(rgb_from_ycrcb_equalized.astype(np.uint8))
    plt.title('Equalização YCrCb (apenas Y)')
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.imshow(rgb_from_lab_equalized.astype(np.uint8))
    plt.title('Equalização LAB (apenas L)')
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.imshow(rgb_from_ycrcb_clahe.astype(np.uint8))
    plt.title('CLAHE em YCrCb (apenas Y)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
   
    print("\nDISCUSSÃO: Por que não aplicar equalização de histograma diretamente em RGB?")
    print("1. Mudança de cor: A equalização de cada canal RGB separadamente pode alterar significativamente o balanço de cores.")
    print("2. Correlação entre canais: Os canais RGB são altamente correlacionados - mudar um sem os outros distorce a cor.")
    print("3. Separação de informações: Em espaços como YCrCb e LAB, a luminância (Y/L) é separada da informação de cor.")
    print("4. Percepção humana: Nossa visão é mais sensível a mudanças de luminância do que de cor.")
    print("5. Preservação de cor: Equalizar apenas o canal de luminância preserva as relações de cor da imagem original.")
    print("6. Resultados naturais: Métodos como equalização em Y (YCrCb) ou L (LAB) produzem resultados mais naturais.")

def show_help():
    """Display help information."""
    print("Uso: python task-sala02.py <número_da_tarefa>")
    print("Tarefas disponíveis:")
    print("  1 - Histogramas de cores para imagens RGB")
    print("  2 - Visualização dos canais de cores individuais")
    print("  3 - Conversão entre espaços de cores")
    print("  4 - Comparação dos efeitos de desfoque em RGB vs HSV")
    print("  5 - Aplicação de filtros de detecção de bordas")
    print("  6 - Filtragem passa-alta e passa-baixa no domínio da frequência")
    print("  7 - Visualização e manipulação de planos de bits")
    print("  8 - Segmentação de objetos baseada em cores usando limiarização HSV")
    print("  9 - Conversão e visualização de imagens no espaço de cores NTSC (YIQ)")
    print("  10 - Aprimoramento de imagem colorida com equalização de histograma")

def main():
    """Main function to run the appropriate task based on command line argument."""
    if len(sys.argv) != 2:
        show_help()
        return
    
    try:
        task_number = int(sys.argv[1])
        
       
        tasks = {
            1: task1,
            2: task2,
            3: task3,
            4: task4,
            5: task5,
            6: task6,
            7: task7,
            8: task8,
            9: task9,
            10: task10
        }
        
        if task_number in tasks:
            tasks[task_number]()
        else:
            print(f"Erro: Tarefa {task_number} não encontrada.")
            show_help()
    
    except ValueError as e:
        print("Erro: O argumento deve ser um número inteiro.")
        print("Detalhes do erro:")
        traceback.print_exc()
        show_help()

if __name__ == "__main__":
    main()




