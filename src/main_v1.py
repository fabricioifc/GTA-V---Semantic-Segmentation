import os
import cv2
import torch
import numpy as np
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
DATASET_URL = "http://200.135.55.29:8888/GTA-V-SID.zip"
DATASET_ZIP = os.path.basename(DATASET_URL)
DATASET_DIR = "dataset"
DATASET_PATH = os.path.join(DATASET_DIR, "GTA-V-SID", "500x500")
# Pasta onde tudo será salvo
OUTPUT_DIR = "results" 
os.makedirs(OUTPUT_DIR, exist_ok=True) # Cria a pasta se ela não existir

IMAGE_SIZE = 256
BATCH_SIZE = 4
EPOCHS = 30
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def baixar_dataset():
    if os.path.isdir(DATASET_PATH):
        return

    os.makedirs(DATASET_DIR, exist_ok=True)

    if not os.path.isfile(DATASET_ZIP):
        urllib.request.urlretrieve(DATASET_URL, DATASET_ZIP)

    with zipfile.ZipFile(DATASET_ZIP, "r") as zip_ref:
        zip_ref.extractall(DATASET_DIR)

    if os.path.isfile(DATASET_ZIP):
        os.remove(DATASET_ZIP)

    print("Dataset baixado.")



class GTADataset(Dataset):
    def __init__(self, path):
        self.img_dir = os.path.join(path, "slice")
        self.mask_dir = os.path.join(path, "label")
        self.images = sorted(os.listdir(self.img_dir))
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, 0)
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.uint8)
        return self.transform(image), torch.tensor(mask, dtype=torch.long)

class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.down1 = DoubleConv(3, 64); self.down2 = DoubleConv(64, 128); self.pool = nn.MaxPool2d(2)
        self.middle = DoubleConv(128, 256)
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2); self.conv1 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2); self.conv2 = DoubleConv(128, 64)
        self.final = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        d1 = self.down1(x); d2 = self.down2(self.pool(d1))
        m = self.middle(self.pool(d2))
        u1 = self.conv1(torch.cat([self.up1(m), d2], dim=1))
        u2 = self.conv2(torch.cat([self.up2(u1), d1], dim=1))
        return self.final(u2)

# =========================================================
# 2. FUNÇÕES COM SALVAMENTO EM ARQUIVO
# =========================================================

def plot_training(history):
    """Gera e salva o gráfico de perda"""
    plt.figure(figsize=(10, 5))
    plt.plot(history, label='Treino')
    plt.title("Histórico de Perda (Loss)")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.grid(True)
    
    save_path = os.path.join(OUTPUT_DIR, "loss_plot.png")
    plt.savefig(save_path)
    plt.close() # Fecha para não sobrecarregar a memória
    print(f"Gráfico de loss salvo em: {save_path}")

def save_predictions(dataset, model, num_samples=5):
    """Realiza predições aleatórias e salva as imagens comparativas"""
    model.eval()
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        image, mask = dataset[idx]
        with torch.no_grad():
            pred = model(image.unsqueeze(0).to(DEVICE))
            pred = torch.argmax(pred, dim=1).squeeze().cpu().numpy()

        img_np = image.permute(1, 2, 0).numpy()
        
        # Criando o painel visual
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img_np); axes[0].set_title("Original")
        axes[1].imshow(mask, cmap='gray'); axes[1].set_title("Gabarito (Ground Truth)")
        axes[2].imshow(pred, cmap='gray'); axes[2].set_title("Predição da Rede")
        
        for ax in axes: ax.axis("off")
        
        save_path = os.path.join(OUTPUT_DIR, f"result_sample_{idx}.png")
        plt.savefig(save_path)
        plt.close()
    print(f"{num_samples} exemplos de predição salvos em: {OUTPUT_DIR}")

def train_model(model, loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    history = []

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for images, masks in loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        history.append(avg_loss)
        print(f"Época [{epoch+1}/{EPOCHS}] - Loss: {avg_loss:.4f}")
    return history

def model_summary(model_path):
    """Gera um resumo textual dos parâmetros e salva em um .txt"""
    model_data = torch.load(model_path, map_location="cpu")
    summary_path = os.path.join(OUTPUT_DIR, "model_summary.txt")
    
    with open(summary_path, "w") as f:
        f.write(f"{'Camada':<45} {'Shape':<20} {'Parâmetros'}\n")
        f.write("-" * 80 + "\n")
        total = 0
        for key, value in model_data.items():
            params = value.numel()
            total += params
            f.write(f"{key:<45} {str(tuple(value.shape)):<20} {params}\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total de Parâmetros: {total:,}\n")
    
    print(f"Resumo do modelo salvo em: {summary_path}")

# =========================================================
# 3. EXECUÇÃO
# =========================================================
if __name__ == "__main__":
    baixar_dataset()

    # Imprimir os hiperparâmetros e caminhos para conferência
    print("Configurações:")
    print(f"Dataset Path: {DATASET_PATH}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Image Size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Device: {DEVICE}\n")

    # 1. Carrega o dataset completo
    dataset = GTADataset(DATASET_PATH)

    # 2. Define os tamanhos (80% treino, 20% teste)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    # 3. Divide aleatoriamente
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # 4. Cria DataLoaders específicos
    # O de treino precisa de shuffle=True para misturar os dados a cada época
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # O de teste não precisa de shuffle, serve apenas para avaliação
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Imagens de Treino: {len(train_dataset)}")
    print(f"Imagens de Teste: {len(test_dataset)}")

    # 5. Inicializa e treina o modelo apenas com os dados de TREINO
    model = UNet(n_classes=2).to(DEVICE)
    history = train_model(model, train_loader)

    # Salvar Pesos
    weights_path = os.path.join(OUTPUT_DIR, "unet_gta.pth")
    torch.save(model.state_dict(), weights_path)

    # Gerar e Salvar arquivos de análise
    plot_training(history)
    model_summary(weights_path)
    save_predictions(test_dataset, model, num_samples=5)
    
    print("\nProcesso concluído! Verifique a pasta:", OUTPUT_DIR)
    