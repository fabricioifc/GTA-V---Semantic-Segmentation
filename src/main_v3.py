import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import zipfile
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import sys

# =========================================================
# 1. CONFIGURAÇÕES E HIPERPARÂMETROS
# =========================================================
DATASET_URL = "http://200.135.55.29:8888/GTA-V-SID.zip"
DATASET_ZIP = os.path.basename(DATASET_URL)
DATASET_DIR = "dataset"
DATASET_PATH = os.path.join(DATASET_DIR, "GTA-V-SID", "500x500")
OUTPUT_DIR = "results_v3" 
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_SIZE = 256
BATCH_SIZE = 8 # Aumentado para melhor proveito do Batch Normalization
EPOCHS = 30
LEARNING_RATE = 0.0005 # Reduzido levemente para estabilidade com Adam
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

# =========================================================
# 2. DATASET COM DATA AUGMENTATION
# =========================================================
class GTADataset(Dataset):
    def __init__(self, path, augment=False):
        self.img_dir = os.path.join(path, "slice")
        self.mask_dir = os.path.join(path, "label")
        self.images = sorted(os.listdir(self.img_dir))
        self.augment = augment
        self.to_tensor = transforms.ToTensor()

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])
        
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, 0)
        
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)

        # DATA AUGMENTATION MANUAL (Sincronizado entre imagem e máscara)
        if self.augment:
            if np.random.random() > 0.5:
                image = cv2.flip(image, 1) # Flip Horizontal
                mask = cv2.flip(mask, 1)
            
            # Brilho aleatório (apenas na imagem)
            if np.random.random() > 0.5:
                factor = np.random.uniform(0.8, 1.2)
                image = (image * factor).clip(0, 255).astype(np.uint8)

        mask = (mask > 0).astype(np.uint8)
        return self.to_tensor(image), torch.tensor(mask, dtype=torch.long)

# # =========================================================
# # 3. ARQUITETURA ATTENTION U-NET (BatchNorm + Dropout + Attention)
# # =========================================================

# class DoubleConv(nn.Module):
#     """Conv -> BatchNorm -> ReLU -> Dropout -> Conv -> BatchNorm -> ReLU"""
#     def __init__(self, in_c, out_c):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_c, out_c, 3, padding=1),
#             nn.BatchNorm2d(out_c), # Estabiliza o treino
#             nn.ReLU(inplace=True),
#             nn.Dropout2d(0.1),     # Previne Overfitting
#             nn.Conv2d(out_c, out_c, 3, padding=1),
#             nn.BatchNorm2d(out_c),
#             nn.ReLU(inplace=True)
#         )
#     def forward(self, x): return self.conv(x)

# class AttentionGate(nn.Module):
#     """Portão de Atenção para focar em áreas relevantes da imagem"""
#     def __init__(self, F_g, F_l, F_int):
#         super().__init__()
#         self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, kernel_size=1), nn.BatchNorm2d(F_int))
#         self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, kernel_size=1), nn.BatchNorm2d(F_int))
#         self.psi = nn.Sequential(nn.Conv2d(F_int, 1, kernel_size=1), nn.BatchNorm2d(1), nn.Sigmoid())
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, g, x):
#         g1 = self.W_g(g)
#         x1 = self.W_x(x)
#         psi = self.relu(g1 + x1)
#         psi = self.psi(psi)
#         return x * psi

# class AttentionUNet(nn.Module):
#     def __init__(self, n_classes=2):
#         super().__init__()
#         self.pool = nn.MaxPool2d(2)
        
#         # Encoder
#         self.d1 = DoubleConv(3, 64)
#         self.d2 = DoubleConv(64, 128)
#         self.middle = DoubleConv(128, 256)

#         # Decoder + Attention
#         self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
#         self.att1 = AttentionGate(F_g=128, F_l=128, F_int=64)
#         self.c1 = DoubleConv(256, 128)

#         self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
#         self.att2 = AttentionGate(F_g=64, F_l=64, F_int=32)
#         self.c2 = DoubleConv(128, 64)

#         self.final = nn.Conv2d(64, n_classes, kernel_size=1)

#     def forward(self, x):
#         s1 = self.d1(x)
#         s2 = self.d2(self.pool(s1))
#         m  = self.middle(self.pool(s2))

#         u1 = self.up1(m)
#         a1 = self.att1(g=u1, x=s2)
#         u1 = self.c1(torch.cat([u1, a1], dim=1))

#         u2 = self.up2(u1)
#         a2 = self.att2(g=u2, x=s1)
#         u2 = self.c2(torch.cat([u2, a2], dim=1))
        
#         return self.final(u2)

# =========================================================
# 4. FUNÇÕES DE PERDA (DICE) E MÉTRICAS (IOU)
# =========================================================

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        probs = F.softmax(logits, dim=1)
        # Pega probabilidade da classe positiva (1)
        probs = probs[:, 1, :, :]
        target = target.float()
        
        intersection = (probs * target).sum()
        dice = (2. * intersection + self.smooth) / (probs.sum() + target.sum() + self.smooth)
        return 1 - dice

def calculate_iou(outputs, masks):
    """Calcula Intersection over Union (Jaccard Index)"""
    preds = torch.argmax(outputs, dim=1)
    inter = (preds & masks).float().sum((1, 2))
    union = (preds | masks).float().sum((1, 2))
    iou = (inter + 1e-6) / (union + 1e-6)
    return iou.mean().item()

# =========================================================
# 5. LOOP DE TREINAMENTO E VALIDAÇÃO
# =========================================================

def train_engine(model, train_loader, val_loader):
    ce_loss = nn.CrossEntropyLoss()
    dice_loss = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    history = {"train_loss": [], "val_iou": []}

    for epoch in range(EPOCHS):
        model.train()
        train_running_loss = 0
        
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            
            out = model(imgs)['out']
            # Combo Loss: 50% CE + 50% Dice
            loss = (0.5 * ce_loss(out, masks)) + (0.5 * dice_loss(out, masks))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_running_loss += loss.item()

        # Validação
        model.eval()
        val_iou = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                out = model(imgs)['out']
                val_iou += calculate_iou(out, masks)
        
        avg_train_loss = train_running_loss / len(train_loader)
        avg_val_iou = val_iou / len(val_loader)
        
        history["train_loss"].append(avg_train_loss)
        history["val_iou"].append(avg_val_iou)
        
        print(f"Época {epoch+1}/{EPOCHS} -> Loss: {avg_train_loss:.4f} | Val IoU: {avg_val_iou:.4f}")

    return history


def save_predictions(dataset, model, num_samples=5):
    """Realiza predições aleatórias e salva as imagens comparativas"""
    model.eval()
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        image, mask = dataset[idx]
        with torch.no_grad():
            # unsqueeze para adicionar dimensão de batch (1, CANAIS, HEIGHT, WIDTH).
            pred = model(image.unsqueeze(0).to(DEVICE))['out']
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

def evaluate_test_set(model, test_loader):
    """Avalia o modelo no conjunto de teste e calcula IoU médio"""
    model.eval()
    test_iou = 0
    with torch.no_grad():
        for imgs, masks in test_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            out = model(imgs)['out']
            test_iou += calculate_iou(out, masks)
    
    avg_test_iou = test_iou / len(test_loader)
    print(f"\n{'='*50}")
    print(f"Test Set IoU: {avg_test_iou:.4f}")
    print(f"{'='*50}\n")
    return avg_test_iou

def plot_training(history):
    """Plota Loss e Val IoU no mesmo gráfico"""
    plt.figure(figsize=(10, 6))
    plt.title("Histórico de Treinamento")
    plt.xlabel("Época")
    plt.ylabel("Valor")
    plt.plot(history["train_loss"], label="Loss", linewidth=2)
    plt.plot(history["val_iou"], label="Val IoU", linewidth=2)
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(OUTPUT_DIR, "training_history.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Gráficos de treinamento salvos em: {save_path}")

    
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
# 6. EXECUÇÃO
# =========================================================
if __name__ == "__main__":
    # se receber o parametro "--test" do terminal, roda apenas a parte de predição para validar o modelo
    test_param = len(sys.argv) > 1 and sys.argv[1] == "--test"
    if test_param:
        model = AttentionUNet(n_classes=2).to(DEVICE)
        model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "attention_unet_gta.pth"), map_location=DEVICE))
        dataset = GTADataset(DATASET_PATH, augment=False)
        save_predictions(dataset, model, num_samples=5)
        exit()
    baixar_dataset()
    dataset = GTADataset(DATASET_PATH, augment=True)
    
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # model = AttentionUNet(n_classes=2).to(DEVICE)
    # usar um modelo pré-treinado pode ajudar a acelerar o processo e melhorar a performance, especialmente com um dataset pequeno como o GTA-V-SID
    from torchvision.models.segmentation import deeplabv3_resnet50
    model = deeplabv3_resnet50(pretrained=True)
    model.classifier[4] = nn.Conv2d(256, 2, kernel_size=1) # Ajusta a última camada para 2 classes
    model = model.to(DEVICE)
    
    print(f"Iniciando Treino em: {DEVICE}")
    history = train_engine(model, train_loader, val_loader)

    # Salvar Pesos
    weights_path = os.path.join(OUTPUT_DIR, "deeplabv3_resnet50_gta.pth")
    torch.save(model.state_dict(), weights_path)

    # Avaliar no conjunto de teste
    evaluate_test_set(model, test_loader)

    # Gerar e Salvar arquivos de análise
    plot_training(history)
    model_summary(weights_path)
    save_predictions(test_ds, model, num_samples=5)
    
    print("\nProcesso concluído! Verifique a pasta:", OUTPUT_DIR)
    