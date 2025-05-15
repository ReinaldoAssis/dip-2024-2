import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report

# Configuração de diretórios
TRAIN_CSV = 'train.csv'
TEST_CSV = 'test.csv'
IMAGE_DIR = 'lego/'  # Ajuste para o caminho correto das imagens

# Função para carregar e pré-processar imagens
def load_and_preprocess(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Erro ao carregar imagem: {image_path}")
        return None
    
    # Converter para RGB (OpenCV carrega como BGR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb

# Função para visualizar imagens
def visualize_image(img, title="Imagem"):
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Função para segmentar a imagem em regiões específicas
def segment_lego_parts(img):
    # Converter para HSV para melhor segmentação de cores
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    # Definir intervalos de cores para as partes principais do Lego
    # Cabeça (geralmente amarela)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Corpo (geralmente vermelho, azul ou outra cor)
    # Para vermelho (que cruza o limite em HSV)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    
    # Para azul
    lower_blue = np.array([90, 100, 100])
    upper_blue = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Para a cor preta (chapéu, etc.)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 50])
    black_mask = cv2.inRange(hsv, lower_black, upper_black)
    
    # Combinar máscaras para cores de corpo
    body_mask = cv2.bitwise_or(red_mask, blue_mask)
    
    # Aplicar operações morfológicas para melhorar a segmentação
    kernel = np.ones((5,5), np.uint8)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
    body_mask = cv2.morphologyEx(body_mask, cv2.MORPH_CLOSE, kernel)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel)
    
    return {
        'head_mask': yellow_mask,
        'body_mask': body_mask,
        'dark_mask': black_mask,
        'hsv': hsv
    }

# Função para detectar se o chapéu está faltando
def detect_no_hat(img, masks):
    # Obter a região superior da imagem onde o chapéu deveria estar
    height, width = img.shape[:2]
    
    # Focamos na região superior da imagem
    top_region = masks['dark_mask'][:height//4, :]
    
    # Verificar se há suficientes pixels pretos na região superior
    hat_pixels = cv2.countNonZero(top_region)
    hat_threshold = width * height // 40  # Ajustar conforme necessário
    
    return 1 if hat_pixels < hat_threshold else 0

# Função para detectar se o rosto está faltando
def detect_no_face(img, masks):
    # Converter para tons de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Primeiro, localizamos a cabeça usando a máscara amarela
    head_mask = masks['head_mask']
    
    # Se não houver cabeça, consideramos que também não há rosto
    if cv2.countNonZero(head_mask) < 100:
        return 1
    
    # Encontrar a região da cabeça
    contours, _ = cv2.findContours(head_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 1
    
    # Pegar o maior contorno (presumivelmente a cabeça)
    head_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(head_contour)
    
    # Extrair a região da cabeça
    head_region = gray[y:y+h, x:x+w]
    
    # Aplicar detecção de bordas para encontrar características faciais
    edges = cv2.Canny(head_region, 50, 150)
    
    # Contar o número de bordas na região da face
    edges_count = cv2.countNonZero(edges)
    
    # Se houver poucas bordas, provavelmente não há detalhes faciais
    face_threshold = w * h // 15  # Ajustar conforme necessário
    
    return 1 if edges_count < face_threshold else 0

# Função para detectar se a cabeça está faltando
def detect_no_head(img, masks):
    head_mask = masks['head_mask']
    
    # Contar pixels da cabeça
    head_pixels = cv2.countNonZero(head_mask)
    
    # Definir um limiar com base no tamanho da imagem
    height, width = img.shape[:2]
    head_threshold = width * height // 20  # Ajustar conforme necessário
    
    return 1 if head_pixels < head_threshold else 0

# Função para detectar se as pernas estão faltando
def detect_no_leg(img, masks):
    height, width = img.shape[:2]
    
    # Focar na parte inferior da imagem
    bottom_region = masks['body_mask'][height//2:, :]
    
    # Encontrar contornos nesta região
    contours, _ = cv2.findContours(bottom_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Se não houver contornos significativos na parte inferior, as pernas estão faltando
    if not contours or max([cv2.contourArea(c) for c in contours], default=0) < width * height // 30:
        return 1
    
    # Verificar a distribuição vertical dos contornos
    max_y = 0
    for c in contours:
        _, y, _, h = cv2.boundingRect(c)
        if y + h > max_y:
            max_y = y + h
    
    # Se os contornos não chegarem perto do fundo da imagem, pode ser que as pernas estejam faltando
    if max_y < bottom_region.shape[0] - height//10:
        return 1
    
    return 0

# Função para detectar se o corpo está faltando ou sem textura
def detect_no_body(img, masks):
    # Obter a região do corpo
    height, width = img.shape[:2]
    
    # Focar na região central da imagem onde o corpo deve estar
    mid_region_mask = masks['body_mask'][height//4:3*height//4, :]
    
    # Contar pixels do corpo nesta região
    body_pixels = cv2.countNonZero(mid_region_mask)
    
    # Definir um limiar adequado
    body_threshold = (height//2) * width // 10
    
    # Se houver poucos pixels coloridos no meio, o corpo está faltando
    if body_pixels < body_threshold:
        return 1
    
    # Também podemos verificar a variância das cores no corpo para detectar falta de textura
    # Obter a região original da imagem onde o corpo está
    hsv = masks['hsv']
    body_region = cv2.bitwise_and(hsv, hsv, mask=mid_region_mask)
    
    # Calcular a variância na região do corpo (canal S para saturação)
    non_zero_pixels = body_region[np.where(mid_region_mask > 0)]
    if len(non_zero_pixels) > 0:
        saturation_variance = np.var(non_zero_pixels[:, 1])
        
        # Se a variância for muito baixa, provavelmente não há textura
        if saturation_variance < 50:  # Ajustar conforme necessário
            return 1
    
    return 0

# Função para detectar se as mãos estão faltando
def detect_no_hand(img, masks):
    height, width = img.shape[:2]
    
    # Em um Lego, as mãos geralmente estão nas extremidades laterais da região central
    mid_height = height // 2
    
    # Verificar ambos os lados para as mãos
    left_side = masks['body_mask'][mid_height-height//6:mid_height+height//6, :width//4]
    right_side = masks['body_mask'][mid_height-height//6:mid_height+height//6, 3*width//4:]
    
    # Contar pixels em cada lado
    left_pixels = cv2.countNonZero(left_side)
    right_pixels = cv2.countNonZero(right_side)
    
    # Definir limiares para cada lado
    side_threshold = (height//3) * (width//4) // 10
    
    # Se algum dos lados não tiver pixels suficientes, uma mão pode estar faltando
    if left_pixels < side_threshold or right_pixels < side_threshold:
        return 1
    
    return 0

# Função para detectar se os braços estão faltando
def detect_no_arm(img, masks):
    height, width = img.shape[:2]
    
    # Os braços geralmente estão nas laterais da região central, um pouco mais extensos que as mãos
    upper_mid = height // 3
    lower_mid = 2 * height // 3
    
    # Verificar ambos os lados para os braços
    left_arm = masks['body_mask'][upper_mid:lower_mid, :width//3]
    right_arm = masks['body_mask'][upper_mid:lower_mid, 2*width//3:]
    
    # Contar pixels em cada lado
    left_pixels = cv2.countNonZero(left_arm)
    right_pixels = cv2.countNonZero(right_arm)
    
    # Definir limiares para cada lado
    arm_threshold = (lower_mid - upper_mid) * (width//3) // 8
    
    # Se algum dos lados não tiver pixels suficientes, um braço pode estar faltando
    if left_pixels < arm_threshold or right_pixels < arm_threshold:
        return 1
    
    return 0

# Função principal para detectar todos os defeitos
def detect_defects(img):
    """
    Função principal que combina todas as detecções de defeitos
    """
    # Segmentar a imagem em partes relevantes
    masks = segment_lego_parts(img)
    
    # Detectar cada tipo de defeito
    no_hat = detect_no_hat(img, masks)
    no_face = detect_no_face(img, masks)
    no_head = detect_no_head(img, masks)
    no_leg = detect_no_leg(img, masks)
    no_body = detect_no_body(img, masks)
    no_hand = detect_no_hand(img, masks)
    no_arm = detect_no_arm(img, masks)
    
    # Determinar se há algum defeito
    has_defect = 1 if (no_hat or no_face or no_head or no_leg or no_body or no_hand or no_arm) else 0
    
    # Retornar resultados
    return {
        'has_defect': has_defect,
        'no_hat': no_hat,
        'no_face': no_face,
        'no_head': no_head,
        'no_leg': no_leg,
        'no_body': no_body,
        'no_hand': no_hand,
        'no_arm': no_arm
    }

# Função para processar todas as imagens
def process_images(image_dir, df):
    results = []
    
    for index, row in df.iterrows():
        example_id = row['example_id']
        image_path = os.path.join(image_dir, f"{example_id}.jpg")  # Ajuste a extensão se necessário
        
        # Carregar e processar a imagem
        img = load_and_preprocess(image_path)
        if img is None:
            continue
        
        # Detectar defeitos
        defects = detect_defects(img)
        
        # Adicionar ID da imagem
        defects['example_id'] = example_id
        
        results.append(defects)
        
        # Opcional: mostrar progresso
        if index % 10 == 0:
            print(f"Processadas {index} imagens...")
    
    return pd.DataFrame(results)

# Função para calcular métricas no conjunto de treinamento
def evaluate_results(pred_df, true_df):
    # Mesclar os dataframes nas colunas de interesse
    merged = pred_df.merge(true_df, on='example_id', suffixes=('_pred', '_true'))
    
    # Calcular acurácia para cada tipo de defeito
    metrics = {}
    
    for defect in ['has_defect', 'no_hat', 'no_face', 'no_head', 'no_leg', 'no_body', 'no_hand', 'no_arm']:
        true_col = f"{defect}_true" if defect in true_df.columns else defect
        pred_col = f"{defect}_pred" if defect in pred_df.columns else defect
        
        if true_col in merged.columns and pred_col in merged.columns:
            accuracy = accuracy_score(merged[true_col], merged[pred_col])
            metrics[defect] = accuracy
    
    return metrics

# Função para ajustar os parâmetros com base nos resultados do conjunto de treinamento
def tune_parameters(train_df, image_dir):
    # Aqui você poderia implementar um loop para testar diferentes limiares e parâmetros
    # Por simplicidade, vamos pular esta etapa neste exemplo
    print("Ajuste de parâmetros: usando valores padrão")
    
    # Você poderia retornar os melhores parâmetros encontrados
    return {}

# Função para visualizar os resultados da detecção
def visualize_detection(img, defects):
    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    
    status = "COMPLIANT" if defects['has_defect'] == 0 else "NON-COMPLIANT"
    title = f"Status: {status}"
    
    if defects['has_defect'] == 1:
        defect_types = []
        if defects['no_hat'] == 1: defect_types.append("NO_HAT")
        if defects['no_face'] == 1: defect_types.append("NO_FACE")
        if defects['no_head'] == 1: defect_types.append("NO_HEAD")
        if defects['no_leg'] == 1: defect_types.append("NO_LEG")
        if defects['no_body'] == 1: defect_types.append("NO_BODY")
        if defects['no_hand'] == 1: defect_types.append("NO_HAND")
        if defects['no_arm'] == 1: defect_types.append("NO_ARM")
        
        title += "\nDefeitos: " + ", ".join(defect_types)
    
    plt.title(title)
    plt.axis('off')
    plt.show()

# Função principal que orquestra todo o processo
def main():
    print("Carregando dados...")
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    
    print(f"Dados de treinamento: {train_df.shape}")
    print(f"Dados de teste: {test_df.shape}")
    
    # Opcional: visualizar algumas imagens de exemplo
    if False:  # Defina como True para visualizar
        for i in range(3):
            example_id = train_df.iloc[i]['example_id']
            image_path = os.path.join(IMAGE_DIR, f"{example_id}.jpg")
            img = load_and_preprocess(image_path)
            if img is not None:
                has_defect = train_df.iloc[i]['has_defect']
                title = "COMPLIANT" if has_defect == 0 else "NON-COMPLIANT"
                visualize_image(img, title)
    
    # Ajustar parâmetros (opcional)
    params = tune_parameters(train_df, IMAGE_DIR)
    
    print("Processando conjunto de treinamento...")
    train_results = process_images(IMAGE_DIR, train_df)
    
    # Avaliar resultados no conjunto de treinamento
    if 'has_defect' in train_df.columns:
        metrics = evaluate_results(train_results, train_df)
        print("\nMétricas no conjunto de treinamento:")
        for defect, accuracy in metrics.items():
            print(f"{defect}: {accuracy:.4f}")
    
    print("\nProcessando conjunto de teste...")
    test_results = process_images(IMAGE_DIR, test_df)
    
    # Preparar envio
    print("Preparando arquivo de submissão...")
    
    # Selecionar apenas as colunas necessárias para o envio
    submission = test_results[['example_id', 'has_defect', 'no_hat', 'no_face', 
                              'no_head', 'no_leg', 'no_body', 'no_hand', 'no_arm']]
    
    # Verificar se temos resultados para todos os exemplos do conjunto de teste
    missing_examples = set(test_df['example_id']) - set(submission['example_id'])
    if missing_examples:
        print(f"Atenção: {len(missing_examples)} exemplos estão faltando no resultado.")
        
        # Adicionar linhas para exemplos faltantes (todos como não defeituosos por padrão)
        for example_id in missing_examples:
            submission = submission.append({
                'example_id': example_id,
                'has_defect': 0,
                'no_hat': 0,
                'no_face': 0,
                'no_head': 0,
                'no_leg': 0,
                'no_body': 0,
                'no_hand': 0,
                'no_arm': 0
            }, ignore_index=True)
    
    # Salvar o arquivo de submissão
    submission.to_csv('submission.csv', index=False)
    print(f"Arquivo de submissão 'submission.csv' salvo com sucesso!")
    
    # Opcional: Visualizar alguns resultados
    print("\nVisualizando alguns exemplos de detecção...")
    for i in range(min(5, len(test_results))):
        example_id = test_results.iloc[i]['example_id']
        image_path = os.path.join(IMAGE_DIR, f"{example_id}.jpg")
        img = load_and_preprocess(image_path)
        
        if img is not None:
            defects = {
                'has_defect': test_results.iloc[i]['has_defect'],
                'no_hat': test_results.iloc[i]['no_hat'],
                'no_face': test_results.iloc[i]['no_face'],
                'no_head': test_results.iloc[i]['no_head'],
                'no_leg': test_results.iloc[i]['no_leg'],
                'no_body': test_results.iloc[i]['no_body'],
                'no_hand': test_results.iloc[i]['no_hand'],
                'no_arm': test_results.iloc[i]['no_arm']
            }
            
            visualize_detection(img, defects)

if __name__ == "__main__":
    main()