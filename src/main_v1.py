import os
import cv2
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import urllib.request
import zipfile
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

# =========================================================
# 1. CONFIGURAÇÕES
# =========================================================

# URL do dataset hospedado em um servidor externo
DATASET_URL = "http://200.135.55.29:8888/GTA-V-SID.zip"

# Nome do arquivo .zip extraído da URL
DATASET_ZIP = os.path.basename(DATASET_URL)

# Pasta raiz onde o dataset será extraído
DATASET_DIR = "dataset"

# Caminho completo até as imagens 500x500 do dataset GTA-V-SID
DATASET_PATH = os.path.join(DATASET_DIR, "GTA-V-SID", "500x500")

# Pasta onde os resultados (gráficos, predições, pesos) serão salvos
OUTPUT_DIR = "results_v1"
os.makedirs(OUTPUT_DIR, exist_ok=True)  # Cria a pasta caso ainda não exista

# Tamanho para o qual todas as imagens serão redimensionadas (256x256 pixels)
IMAGE_SIZE = 256

# Número de imagens processadas por vez durante o treinamento
# Valores menores usam menos memória RAM/VRAM, mas treinam mais devagar
BATCH_SIZE = 8

# Número de vezes que o modelo verá o dataset completo durante o treino
EPOCHS = 30

# Taxa de aprendizado: controla o quanto os pesos são ajustados a cada passo
# Valores muito altos podem fazer o modelo "pular" o mínimo da loss;
# valores muito baixos tornam o treino lento demais
LEARNING_RATE = 0.001

# Seed global para reprodutibilidade (random state)
# SEED = 42

# Seleciona automaticamente GPU (cuda) se disponível, caso contrário usa CPU
# GPU é muito mais rápida para operações com tensores
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# def set_seed(seed=SEED):
#     """Define seeds dos principais geradores aleatórios usados no experimento."""
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)

#     # Força operações determinísticas quando disponível.
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# =========================================================
# DOWNLOAD DO DATASET
# =========================================================

def baixar_dataset():
    """
    Verifica se o dataset já existe localmente.
    Se não existir, faz o download, extrai o .zip e remove o arquivo comprimido.
    """
    # Se a pasta de imagens já existir, o dataset já foi baixado — não faz nada
    if os.path.isdir(DATASET_PATH):
        return

    os.makedirs(DATASET_DIR, exist_ok=True)

    # Baixa o .zip somente se ele ainda não estiver no disco
    if not os.path.isfile(DATASET_ZIP):
        print("Baixando dataset...")
        urllib.request.urlretrieve(DATASET_URL, DATASET_ZIP)

    # Extrai todo o conteúdo do .zip para a pasta DATASET_DIR
    with zipfile.ZipFile(DATASET_ZIP, "r") as zip_ref:
        zip_ref.extractall(DATASET_DIR)

    # Remove o .zip após a extração para economizar espaço em disco
    if os.path.isfile(DATASET_ZIP):
        os.remove(DATASET_ZIP)

    print("Dataset baixado e extraído com sucesso.")


# =========================================================
# DATASET PERSONALIZADO
# =========================================================

class GTADataset(Dataset):
    """
    Dataset personalizado para o GTA-V-SID.

    O PyTorch exige que datasets implementem três métodos:
      - __init__: inicializa caminhos e transformações
      - __len__:  retorna o número total de amostras
      - __getitem__: retorna uma amostra (imagem + máscara) pelo índice
    """

    def __init__(self, path):
        # Subpasta com as imagens originais (fatias do jogo)
        self.img_dir = os.path.join(path, "slice")

        # Subpasta com as máscaras de segmentação (rótulos/labels)
        self.mask_dir = os.path.join(path, "label")

        # Lista ordenada dos nomes dos arquivos de imagem
        self.images = sorted(os.listdir(self.img_dir))

        # Transformação que converte um array NumPy HxWxC (0-255)
        # para um tensor PyTorch CxHxW (0.0-1.0)
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        # Retorna quantas imagens existem no dataset
        return len(self.images)

    def __getitem__(self, idx):
        # Monta os caminhos completos da imagem e da máscara correspondente
        img_path  = os.path.join(self.img_dir,  self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])

        # Lê a imagem em BGR (padrão do OpenCV) e converte para RGB
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        # Lê a máscara em escala de cinza (0 = fundo, >0 = objeto de interesse)
        mask = cv2.imread(mask_path, 0)

        # Redimensiona a imagem para o tamanho definido nas configurações
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

        # Redimensiona a máscara usando interpolação NEAREST para não criar
        # valores intermediários entre as classes (ex: evita valor 128 entre 0 e 255)
        mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE),
                          interpolation=cv2.INTER_NEAREST)

        # Binariza a máscara: qualquer pixel > 0 vira classe 1, 0 permanece classe 0
        # Resultado: máscara com apenas dois valores — 0 (fundo) e 1 (objeto)
        mask = (mask > 0).astype(np.uint8)

        # Retorna a imagem como tensor float e a máscara como tensor de inteiros (long)
        # long é obrigatório para funcionar com CrossEntropyLoss
        return self.transform(image), torch.tensor(mask, dtype=torch.long)


# =========================================================
# BLOCOS DA ARQUITETURA U-NET
# =========================================================

class DoubleConv(nn.Module):
    """
    Bloco de dupla convolução: Conv → ReLU → Conv → ReLU.

    É o bloco básico da U-Net. Aplicado duas vezes seguidas para
    extrair features mais ricas sem reduzir a resolução espacial.
    padding=1 mantém a mesma largura e altura da entrada.
    """

    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),   # inplace=True economiza memória
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    Implementação simplificada da U-Net para segmentação semântica binária.

    Arquitetura:
      Encoder (descida):  extrai features progressivamente, reduzindo resolução com MaxPool
      Bottleneck (meio):  camada mais profunda com maior número de filtros
      Decoder (subida):   reconstrói a resolução com ConvTranspose2d + skip connections

    As skip connections concatenam features do encoder com as do decoder,
    permitindo que detalhes espaciais perdidos no pooling sejam recuperados.
    """

    def __init__(self, n_classes):
        super().__init__()

        # ----- ENCODER -----
        # down1: 3 canais RGB → 64 feature maps
        self.down1 = DoubleConv(3, 64)
        # down2: 64 → 128 feature maps (resolução já reduzida pelo pool)
        self.down2 = DoubleConv(64, 128)
        # MaxPool2d(2) divide largura e altura pela metade a cada chamada
        self.pool = nn.MaxPool2d(2)

        # ----- BOTTLENECK -----
        # Camada do meio: maior profundidade, menor resolução espacial
        self.middle = DoubleConv(128, 256)

        # ----- DECODER -----
        # ConvTranspose2d dobra a resolução espacial (operação inversa ao pool)
        # up1: 256 → 128 canais, depois concatena com skip de down2 (256 total) → DoubleConv para 128
        self.up1   = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(256, 128)  # 128 (up) + 128 (skip) = 256 entradas

        # up2: 128 → 64 canais, depois concatena com skip de down1 (128 total) → DoubleConv para 64
        self.up2   = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(128, 64)   # 64 (up) + 64 (skip) = 128 entradas

        # Camada final: convolução 1x1 projeta os 64 canais no número de classes
        # kernel_size=1 não altera resolução, apenas muda a profundidade
        self.final = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # --- Encoder ---
        d1 = self.down1(x)          # Features nível 1 (resolução original)
        d2 = self.down2(self.pool(d1))  # Features nível 2 (resolução /2)

        # --- Bottleneck ---
        m = self.middle(self.pool(d2))  # Features do fundo (resolução /4)

        # --- Decoder com skip connections ---
        # Sobe a resolução e concatena com a skip connection do mesmo nível do encoder
        u1 = self.conv1(torch.cat([self.up1(m), d2], dim=1))   # resolução /2
        u2 = self.conv2(torch.cat([self.up2(u1), d1], dim=1))  # resolução original

        # Produz o mapa de logits: shape (batch, n_classes, H, W)
        return self.final(u2)


# =========================================================
# 2. FUNÇÕES COM SALVAMENTO EM ARQUIVO
# =========================================================

def plot_training(history):
    """
    Plota a curva de perda (loss) ao longo das épocas e salva como imagem PNG.
    Útil para diagnosticar overfitting, underfitting ou instabilidade no treino.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(history, label='Treino')
    plt.title("Histórico de Perda (Loss)")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.grid(True)

    save_path = os.path.join(OUTPUT_DIR, "loss_plot.png")
    plt.savefig(save_path)
    plt.close()  # Fecha a figura para liberar memória
    print(f"Gráfico de loss salvo em: {save_path}")


def save_predictions(dataset, model, num_samples=5):
    """
    Seleciona amostras aleatórias do dataset de teste, gera as predições
    e salva painéis comparativos: Imagem Original | Ground Truth | Predição.
    """
    model.eval()  # Desativa dropout e batch norm em modo de avaliação

    # Escolhe índices aleatórios sem repetição
    indices = np.random.choice(len(dataset), num_samples, replace=False)

    for i, idx in enumerate(indices):
        image, mask = dataset[idx]

        with torch.no_grad():
            # torch.no_grad() desativa o cálculo de gradientes — economiza memória
            # e acelera a inferência, pois não precisamos de backprop aqui

            # unsqueeze(0) adiciona a dimensão de batch: (C, H, W) → (1, C, H, W)
            pred = model(image.unsqueeze(0).to(DEVICE))

            # argmax na dimensão de classes → índice da classe mais provável por pixel
            # squeeze() remove a dimensão de batch: (1, H, W) → (H, W)
            pred = torch.argmax(pred, dim=1).squeeze().cpu().numpy()

        # Converte tensor (C, H, W) para NumPy (H, W, C) para exibição com matplotlib
        img_np = image.permute(1, 2, 0).numpy()

        # Cria um painel com 3 imagens lado a lado
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img_np);  axes[0].set_title("Original")
        axes[1].imshow(mask, cmap='gray'); axes[1].set_title("Gabarito (Ground Truth)")
        axes[2].imshow(pred, cmap='gray'); axes[2].set_title("Predição da Rede")

        for ax in axes:
            ax.axis("off")  # Remove eixos para visualização mais limpa

        save_path = os.path.join(OUTPUT_DIR, f"result_sample_{idx}.png")
        plt.savefig(save_path)
        plt.close()

    print(f"{num_samples} exemplos de predição salvos em: {OUTPUT_DIR}")


def train_model(model, loader):
    """
    Loop principal de treinamento supervisionado.

    A cada época:
      1. Itera sobre todos os batches do loader
      2. Faz a passagem forward (predição)
      3. Calcula a perda com CrossEntropyLoss
      4. Faz backpropagation para calcular gradientes
      5. Atualiza os pesos com o otimizador Adam
    """
    # CrossEntropyLoss é adequada para segmentação multiclasse.
    criterion = nn.CrossEntropyLoss()

    # Adam adapta a taxa de aprendizado para cada parâmetro individualmente,
    # convergindo mais rápido que o SGD clássico na maioria dos casos
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) # ADAM
    # optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9) # SGD com momentum
    

    history = []  # Armazena a loss média de cada época para gerar o gráfico

    for epoch in range(EPOCHS):
        model.train()  # Ativa o modo de treino (habilita dropout, batch norm, etc.)
        total_loss = 0

        for images, masks in loader:
            # Move os dados para o device correto (CPU ou GPU)
            images, masks = images.to(DEVICE), masks.to(DEVICE)

            # Forward pass: o modelo gera as predições
            outputs = model(images)  # Shape: (batch, n_classes, H, W)

            # Calcula a perda entre predições e rótulos reais
            loss = criterion(outputs, masks)

            # Zera os gradientes acumulados do passo anterior
            # (PyTorch acumula gradientes por padrão — é necessário resetar)
            optimizer.zero_grad()

            # Backward pass: calcula os gradientes de todos os parâmetros
            loss.backward()

            # Atualiza os pesos usando os gradientes calculados
            optimizer.step()

            total_loss += loss.item()  # .item() converte tensor escalar para float Python

        # Média da loss sobre todos os batches da época
        avg_loss = total_loss / len(loader)
        history.append(avg_loss)
        print(f"Época [{epoch+1}/{EPOCHS}] - Loss: {avg_loss:.4f}")

    return history


def model_summary(model_path):
    """
    Carrega os pesos salvos e gera um resumo textual com nome de cada camada,
    shape dos tensores e número de parâmetros. Salva em um arquivo .txt.
    """
    # Carrega o state_dict (dicionário de pesos) diretamente na CPU
    model_data = torch.load(model_path, map_location="cpu")
    summary_path = os.path.join(OUTPUT_DIR, "model_summary.txt")

    with open(summary_path, "w") as f:
        f.write(f"{'Camada':<45} {'Shape':<20} {'Parâmetros'}\n")
        f.write("-" * 80 + "\n")
        total = 0
        for key, value in model_data.items():
            # numel() retorna o número total de elementos no tensor
            params = value.numel()
            total += params
            f.write(f"{key:<45} {str(tuple(value.shape)):<20} {params}\n")
        f.write("-" * 80 + "\n")
        # Exibe o total formatado com separador de milhar para facilitar a leitura
        f.write(f"Total de Parâmetros: {total:,}\n")

    print(f"Resumo do modelo salvo em: {summary_path}")

# =========================================================
# 3. EXECUÇÃO PRINCIPAL
# =========================================================

if __name__ == "__main__":
    # set_seed(SEED)

    # Garante que o dataset está disponível antes de qualquer operação
    baixar_dataset()

    # Exibe as configurações do experimento para facilitar a reprodução
    print("Configurações:")
    print(f"  Dataset Path:      {DATASET_PATH}")
    print(f"  Output Directory:  {OUTPUT_DIR}")
    print(f"  Image Size:        {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"  Batch Size:        {BATCH_SIZE}")
    print(f"  Epochs:            {EPOCHS}")
    print(f"  Learning Rate:     {LEARNING_RATE}")
    # print(f"  Seed:              {SEED}")
    print(f"  Device:            {DEVICE}\n")

    # 1. Carrega o dataset completo (todas as imagens + máscaras)
    dataset = GTADataset(DATASET_PATH)

    # 2. Calcula os tamanhos de cada split
    train_size = int(0.8 * len(dataset))   # 80% para treino
    test_size  = len(dataset) - train_size  # 20% para teste

    # 3. Divide aleatoriamente o dataset em treino e teste
    # random_split garante que não há sobreposição entre os conjuntos
    # split_generator = torch.Generator().manual_seed(SEED)
    train_dataset, test_dataset = random_split(
        dataset,
        [train_size, test_size],
        # generator=split_generator
    )

    # 4. Cria os DataLoaders para cada split
    # shuffle=True no treino: embaralha os dados a cada época, evitando
    # que o modelo aprenda a ordem das amostras em vez dos padrões visuais
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # shuffle=False no teste: a ordem não importa, só queremos avaliar
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Imagens de Treino: {len(train_dataset)}")
    print(f"Imagens de Teste:  {len(test_dataset)}")

    # 5. Instancia a U-Net com 2 classes (fundo e objeto) e move para o device
    model = UNet(n_classes=2).to(DEVICE)

    # 6. Treina o modelo usando apenas os dados de treino
    history = train_model(model, train_loader)

    # 7. Salva os pesos aprendidos para reutilização futura
    # state_dict contém apenas os parâmetros (pesos e biases), não a arquitetura
    weights_path = os.path.join(OUTPUT_DIR, "unet_gta.pth")
    torch.save(model.state_dict(), weights_path)

    # 8. Gera e salva os artefatos de análise do experimento
    plot_training(history)          # Curva de loss
    model_summary(weights_path)     # Resumo textual dos parâmetros
    save_predictions(test_dataset, model, num_samples=5)  # Predições visuais

    print("\nProcesso concluído! Verifique a pasta:", OUTPUT_DIR)