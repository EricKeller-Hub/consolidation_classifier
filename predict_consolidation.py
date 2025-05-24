import os
import numpy as np
import tensorflow as tf
import cv2 # Importa OpenCV para operações de imagem

# --- Configurações ---
# Garanta que o IMG_SIZE seja o mesmo usado no treinamento (225 no seu caso)
IMG_SIZE = 225

def preprocess_image(image_array):
    """
    Pré-processa uma imagem para ser compatível com o modelo treinado.
    Inclui conversão para tons de cinza, equalização de histograma,
    filtro Gaussiano, normalização e redimensionamento.
    """
    # Se a imagem for colorida (3 canais), converte para tons de cinza
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    
    # Converte para uint8 para operações do OpenCV (que esperam valores de 0-255)
    image_array = image_array.astype(np.uint8)
    
    # Aplica equalização de histograma para melhorar contraste
    image_array = cv2.equalizeHist(image_array)
    
    # Aplica filtro Gaussiano para reduzir ruído
    image_array = cv2.GaussianBlur(image_array, (5, 5), 0)
    
    # Normaliza os valores dos pixels para o range [0, 1]
    image_array = image_array.astype(np.float32) / 255.0
    
    # Redimensiona a imagem para o tamanho esperado pelo modelo
    image_array = cv2.resize(image_array, (IMG_SIZE, IMG_SIZE))
    
    # Adiciona a dimensão do canal (para imagens em tons de cinza é 1)
    image_array = np.expand_dims(image_array, axis=-1) # (IMG_SIZE, IMG_SIZE, 1)
    
    # Adiciona a dimensão do batch (para que o modelo possa processar uma única imagem)
    image_array = np.expand_dims(image_array, axis=0)  # (1, IMG_SIZE, IMG_SIZE, 1)

    return image_array

def image_predict(model_path, preprocessed_image_array):
    """
    Carrega o modelo treinado e faz uma predição em uma imagem pré-processada.
    Retorna o valor de probabilidade bruto da predição.
    """
    try:
        modelo = tf.keras.models.load_model(model_path)
        print("Resumo do modelo carregado:")
        modelo.summary()
        predicao = modelo.predict(preprocessed_image_array)
        return predicao
    except Exception as e:
        print(f"Erro ao carregar o modelo ou fazer a predição: {e}")
        return None

# --- Teste da Predição ---
def main():
    # Caminhos para a imagem de teste e o modelo
    # ESTE CAMINHO FOI ATUALIZADO PARA USAR SUA PASTA 'test_data'
    # CERTIFIQUE-SE DE QUE O NOME DO ARQUIVO E A EXTENSÃO ESTÃO EXATOS COMO NO SEU SISTEMA!
    path_imagem_teste = 'C:\\Users\\Keller\\Desktop\\imagem.png' 
    
    # E o caminho do modelo (presumindo que ainda está na pasta 'Mineração')
    path_modelo = 'C:\\Users\\Keller\\Desktop\\Mineração\\last_model.keras' # ou 'last_model.keras'

    # 1. Verificar se os caminhos existem
    if not os.path.exists(path_imagem_teste):
        print(f"Erro: Imagem de teste não encontrada em '{path_imagem_teste}'")
        print("Por favor, verifique o caminho e o nome do arquivo da imagem de teste (incluindo a extensão).")
        return
    if not os.path.exists(path_modelo):
        print(f"Erro: Modelo não encontrado em '{path_modelo}'")
        print("Por favor, verifique o caminho e o nome do arquivo do modelo ('best_model.keras' ou 'last_model.keras').")
        return
    
    # 2. Carregar a imagem original (em tons de cinza)
    # cv2.IMREAD_GRAYSCALE garante que a imagem seja lida como tons de cinza.
    imagem_original = cv2.imread(path_imagem_teste, cv2.IMREAD_GRAYSCALE)

    if imagem_original is None:
        print(f"Erro ao carregar a imagem original: {path_imagem_teste}")
        print("Verifique se o arquivo da imagem está corrompido ou não é um formato de imagem válido.")
        return
    
    print(f"Imagem original carregada com shape: {imagem_original.shape}")
    
    # 3. Pré-processar a imagem
    imagem_preprocessada = preprocess_image(imagem_original)
    print(f"Shape da imagem após pré-processamento para o modelo: {imagem_preprocessada.shape}")

    # 4. Fazer a predição
    predicao_imagem = image_predict(path_modelo, imagem_preprocessada)
    
    if predicao_imagem is not None:
        # A predição é um array de array (ex: [[0.12345]]), então acessamos [0][0]
        probabilidade_normal = predicao_imagem[0][0] 
        print(f"\nPredição (probabilidade bruta): {probabilidade_normal:.4f}")

        # Definir o limiar de decisão
        # Se 0.5 estiver resultando em muitos Falsos Positivos,
        # você pode tentar aumentar este limiar (ex: 0.6, 0.7)
        # para tornar o modelo mais "cauteloso" em prever Consolidação.
        # Faça testes para encontrar o melhor ponto.
        limiar = 0.5 

        # Lembre-se do mapeamento de classes:
        # Class 0 (Consolidation) - probabilidade mais próxima de 0
        # Class 1 (Normal) - probabilidade mais próxima de 1
        
        if probabilidade_normal > limiar:
            print(f"Resultado: Normal (Probabilidade: {probabilidade_normal:.2f} > {limiar})")
        else:
            print(f"Resultado: Consolidation (Probabilidade: {probabilidade_normal:.2f} <= {limiar})")

if __name__ == '__main__':
    main()