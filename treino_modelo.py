import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, roc_curve, auc
import seaborn as sns
import cv2 # Não usado diretamente no fluxo, mas mantido caso precise
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split # Não usado diretamente no fluxo atual, mas mantido

# Configurações globais
IMG_SIZE = 225
BATCH_SIZE = 32
EPOCHS = 50

# Removido a função download_database, pois o dataset já está local.

def prepare_dataset(base_dir):
    """
    Prepara o dataset dividindo em treino (60%), validação (20%) e teste (20%)
    mantendo o balanceamento entre as classes.
    """
    print("Preparando o dataset para Consolidação vs. Normal...")

    # !! MUDANÇA AQUI: Defina os nomes das suas pastas de classes !!
    # As pastas 'consolidation_pure' e 'normal' devem estar diretamente dentro de 'base_dir'
    consolidation_dir = os.path.join(base_dir, 'consolidation_pure')
    normal_dir = os.path.join(base_dir, 'normal')
    
    print(f"\nVerificando diretórios:")
    print(f"Consolidação: {consolidation_dir}")
    print(f"Normal: {normal_dir}")

    # Verificar se os diretórios existem
    if not os.path.exists(consolidation_dir):
        print(f"Erro: Diretório {consolidation_dir} não encontrado!")
        return None, None, None
    if not os.path.exists(normal_dir):
        print(f"Erro: Diretório {normal_dir} não encontrado!")
        return None, None, None
    
    # Função auxiliar para listar arquivos de imagem
    def get_image_files(directory):
        image_files = []
        for file in os.listdir(directory):
            # Adicione outras extensões se suas imagens tiverem (ex: .dcm para DICOM)
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(file)
        return image_files
    
    consolidation_images = get_image_files(consolidation_dir)
    normal_images = get_image_files(normal_dir)
    
    print(f"\nEncontradas:")
    print(f"Imagens de consolidação: {len(consolidation_images)}")
    print(f"Imagens normais: {len(normal_images)}")
    
    if len(consolidation_images) == 0 or len(normal_images) == 0:
        print("Erro: Nenhuma imagem encontrada em um ou mais diretórios!")
        return None, None, None
    
    # Encontrar o número mínimo de imagens entre as classes para balancear o dataset
    min_images = min(len(consolidation_images), len(normal_images))
    print(f"\nUsando {min_images} imagens por classe para manter o balanceamento.")
    
    # Definir os diretórios de destino para os dados divididos
    train_dir = os.path.join(base_dir, 'train_data') 
    val_dir = os.path.join(base_dir, 'val_data')
    test_dir = os.path.join(base_dir, 'test_data')
    
    # Limpar diretórios existentes para garantir uma nova divisão
    # CUIDADO: Isso irá APAGAR as pastas 'train_data', 'val_data', 'test_data'
    # e todo o seu conteúdo antes de recriá-los.
    for split_dir in [train_dir, val_dir, test_dir]:
        if os.path.exists(split_dir):
            shutil.rmtree(split_dir)
        # !! MUDANÇA AQUI: Criar subpastas com os nomes das suas classes para os geradores !!
        os.makedirs(os.path.join(split_dir, 'Consolidation'), exist_ok=True)
        os.makedirs(os.path.join(split_dir, 'Normal'), exist_ok=True)
    
    # Embaralhar as imagens para garantir aleatoriedade na divisão
    np.random.shuffle(normal_images)
    np.random.shuffle(consolidation_images)
    
    # Definir os tamanhos para as divisões (60% treino, 20% validação, 20% teste)
    train_size = int(min_images * 0.6)
    val_size = int(min_images * 0.2)
    # O restante (min_images - train_size - val_size) será para teste
    
    # Função auxiliar para copiar imagens
    def copy_images(images, source_dir, target_class_folder, target_split_dir, start, end):
        copied = 0
        for img in images[start:end]:
            try:
                src = os.path.join(source_dir, img)
                dst = os.path.join(target_split_dir, target_class_folder, img)
                if not os.path.exists(dst): # Copiar apenas se o arquivo ainda não existir
                    shutil.copy2(src, dst)
                    copied += 1
            except Exception as e:
                print(f"Erro ao copiar {img}: {str(e)}")
        return copied
    
    print("\nCopiando imagens para os diretórios de treino, validação e teste...")
    
    # Copiar imagens para os diretórios de destino
    # Normal
    train_normal = copy_images(normal_images, normal_dir, 'Normal', train_dir, 0, train_size)
    val_normal = copy_images(normal_images, normal_dir, 'Normal', val_dir, train_size, train_size + val_size)
    test_normal = copy_images(normal_images, normal_dir, 'Normal', test_dir, train_size + val_size, min_images)
    
    # Consolidation
    train_consolidation = copy_images(consolidation_images, consolidation_dir, 'Consolidation', train_dir, 0, train_size)
    val_consolidation = copy_images(consolidation_images, consolidation_dir, 'Consolidation', val_dir, train_size, train_size + val_size)
    test_consolidation = copy_images(consolidation_images, consolidation_dir, 'Consolidation', test_dir, train_size + val_size, min_images)
    
    print(f"\nDivisão do dataset concluída:")
    print(f"Treino: {train_normal} imagens normais, {train_consolidation} imagens de consolidação")
    print(f"Validação: {val_normal} imagens normais, {val_consolidation} imagens de consolidação")
    print(f"Teste: {test_normal} imagens normais, {test_consolidation} imagens de consolidação")
    
    return train_dir, val_dir, test_dir

def create_model():
    """
    Cria o modelo CNN para classificação binária.
    """
    model = models.Sequential([
        # Primeira camada convolucional
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Segunda camada convolucional
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Terceira camada convolucional
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Camadas densas
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid') # Camada de saída para classificação binária
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy', # Função de perda para classificação binária
        metrics=['accuracy']
    )
    
    return model

def plot_training_history(history):
    """
    Plota os gráficos de acurácia e perda durante o treinamento.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Gráfico de acurácia
    ax1.plot(history.history['accuracy'], label='Treino')
    ax1.plot(history.history['val_accuracy'], label='Validação')
    ax1.set_title('Acurácia durante o treinamento')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Acurácia')
    ax1.legend()
    
    # Gráfico de perda
    ax2.plot(history.history['loss'], label='Treino')
    ax2.plot(history.history['val_loss'], label='Validação')
    ax2.set_title('Perda durante o treinamento')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Perda')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def plot_classification_metrics(y_true, y_pred, class_names, output_path='classification_metrics.png'):
    """
    Plota um gráfico de barras comparando as métricas de precisão, revocação e F1-Score
    para cada classe e as médias macro/ponderada.
    """
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
    macro = precision_recall_fscore_support(y_true, y_pred, average='macro')
    weighted = precision_recall_fscore_support(y_true, y_pred, average='weighted')

    data = {
        'Categoria': [f'{class_names[0]} (Classe 0)', f'{class_names[1]} (Classe 1)', 'Média Macro', 'Média Ponderada'],
        'Precisão': [precision[0], precision[1], macro[0], weighted[0]],
        'Revocação': [recall[0], recall[1], macro[1], weighted[1]],
        'F1-Score': [f1[0], f1[1], macro[2], weighted[2]],
    }
    df = pd.DataFrame(data)
    df_melted = df.melt(id_vars='Categoria', var_name='Métricas', value_name='Pontuação')

    plt.figure(figsize=(10, 5))
    ax = sns.barplot(data=df_melted, x='Categoria', y='Pontuação', hue='Métricas')
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                             ha='center', va='bottom', fontsize=10, color='black', xytext=(0, 3), textcoords='offset points')
    plt.ylim(0, 1.05)
    plt.title('Desempenho do Modelo por Classe e Métrica')
    plt.ylabel('Pontuação')
    plt.xlabel('Categorias')
    plt.legend(title='Métricas')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_roc_curve(y_true, y_score, output_path='roc_curve.png'):
    """
    Plota a Curva ROC (Receiver Operating Characteristic) para classificação binária.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='orange', label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    # !! MUDANÇA CRÍTICA AQUI: Define o caminho base do seu dataset !!
    # Este é o diretório que contém as pastas 'consolidation_pure' e 'normal'
    base_dir = "C:\\Users\\Keller\\Desktop\\Mineração" 
    
    # Preparar o dataset
    train_dir, val_dir, test_dir = prepare_dataset(base_dir)
    
    if train_dir is None or val_dir is None or test_dir is None:
        print("Erro ao preparar o dataset. Encerrando...")
        return

    try:
        # Geração de dados com normalização e aumento de dados
        train_datagen = ImageDataGenerator(
            rescale=1./255, # Normaliza os pixels para o intervalo [0, 1]
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.2,
            shear_range=0.15,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        test_datagen = ImageDataGenerator(rescale=1./255) # Apenas normalização para validação e teste

        # Geradores de dados que carregam as imagens dos diretórios
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(IMG_SIZE, IMG_SIZE),
            color_mode='grayscale', # Imagens em escala de cinza (1 canal)
            batch_size=BATCH_SIZE,
            class_mode='binary', # Para classificação binária (saída 0 ou 1)
            shuffle=True # Embaralha as imagens para o treinamento
        )
        print('Mapeamento de classes:', train_generator.class_indices)
        # O ImageDataGenerator mapeará as pastas 'Consolidation' e 'Normal' para 0 e 1 automaticamente.
        # Por exemplo: {'Consolidation': 0, 'Normal': 1} ou vice-versa.
        # Você deve verificar a saída acima para confirmar qual classe é 0 e qual é 1.

        validation_generator = test_datagen.flow_from_directory(
            val_dir,
            target_size=(IMG_SIZE, IMG_SIZE),
            color_mode='grayscale',
            batch_size=BATCH_SIZE,
            class_mode='binary',
            shuffle=False # Não embaralha para validação e teste
        )

        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=(IMG_SIZE, IMG_SIZE),
            color_mode='grayscale',
            batch_size=BATCH_SIZE,
            class_mode='binary',
            shuffle=False # Não embaralha para avaliação final
        )

        # Cria o modelo CNN
        model = create_model()

        # Callbacks para otimizar o treinamento
        early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
        model_checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True, verbose=1)

        # Treinamento do modelo
        print("\nIniciando o treinamento do modelo...")
        history = model.fit(
            train_generator,
            epochs=EPOCHS,
            validation_data=validation_generator,
            callbacks=[early_stopping, model_checkpoint],
            verbose=1
        )

        # Salvar o modelo final treinado
        model.save('last_model.keras')
        print("Modelo final salvo como 'last_model.keras'.")

        # Avaliação do modelo no conjunto de teste
        print("\nAvaliando o modelo no conjunto de teste...")
        test_loss, test_acc = model.evaluate(test_generator)
        print(f'\nAcurácia no conjunto de teste: {test_acc:.4f}')

        # Realizar previsões no conjunto de teste para calcular métricas detalhadas
        print("\nFazendo previsões no conjunto de teste...")
        predictions = model.predict(test_generator)
        y_pred = (predictions > 0.5).astype(int) # Converte probabilidades para classes binárias (0 ou 1)
        y_true = test_generator.classes # As verdadeiras labels do conjunto de teste

        # Calcular e imprimir o relatório de classificação
        print('\nRelatório de Classificação:')
        # É importante que a ordem de class_names_for_plots corresponda ao mapeamento do gerador.
        # Por exemplo, se o gerador mapeou 'Consolidation': 0 e 'Normal': 1, então class_names_for_plots = ['Consolidation', 'Normal']
        # Se mapeou 'Normal': 0 e 'Consolidation': 1, então class_names_for_plots = ['Normal', 'Consolidation']
        # Verifique a saída de 'train_generator.class_indices' acima para ter certeza.
        
        # !! VERIFIQUE A ORDEM AQUI, baseada no mapeamento do seu gerador !!
        # Exemplo baseado na saída comum: {'Consolidation': 0, 'Normal': 1}
        class_names_for_plots = ['Consolidation', 'Normal'] 

        print(classification_report(y_true, y_pred, target_names=class_names_for_plots))

        # Plotar a matriz de confusão
        print("\nGerando Matriz de Confusão...")
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names_for_plots, yticklabels=class_names_for_plots)
        plt.title('Matriz de Confusão')
        plt.ylabel('Valor Real')
        plt.xlabel('Valor Previsto')
        plt.savefig('confusion_matrix.png')
        plt.close()

        # Plotar o gráfico de métricas por classe
        print("Gerando Gráfico de Métricas de Classificação...")
        plot_classification_metrics(y_true, y_pred, class_names_for_plots)

        # Plotar a curva ROC
        print("Gerando Curva ROC...")
        plot_roc_curve(y_true, predictions)

        print("\nTreinamento e avaliação finalizados. Gráficos e modelo salvos.")

    except Exception as e:
        print(f"Ocorreu um erro durante o treinamento ou avaliação: {str(e)}")

if __name__ == "__main__":
    main()