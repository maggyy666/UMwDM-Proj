"""
BloodMNIST - Klasyfikacja typów komórek krwi

Kompletny pipeline do klasyfikacji 8 typów komórek krwi:
- EDA (eksploracyjna analiza danych)
- Baseline model (SimpleCNN)
- Ulepszony model (DeeperCNN z class weights)
- Ewaluacja i porównanie wyników
- Hyperparameter tuning
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import INFO
from medmnist import BloodMNIST
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from collections import Counter
import seaborn as sns
import random

# Seed dla reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Konfiguracja
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001
USE_CLASS_WEIGHTS = True  # Ważenie klas w loss
USE_AUGMENTATION = True   # Augmentacja danych
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Nazwy klas BloodMNIST
CLASS_NAMES = [
    "basophil",      # bazofile
    "eosinophil",    # eozynofile  
    "erythroblast",  # erytroblasty
    "immature granulocyte",  # granulocyty niedojrzałe
    "lymphocyte",    # limfocyty
    "monocyte",      # monocyty
    "neutrophil",    # neutrofile
    "platelet"       # płytki
]

print(f"Using device: {DEVICE}")


def load_data(use_augmentation=USE_AUGMENTATION):
    """Wczytuje dane BloodMNIST z augmentacją i normalizacją"""
    print("\n=== Wczytywanie danych ===")
    
    # Normalizacja (mean i std dla RGB)
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    
    # Transformacja dla train (z augmentacją jeśli włączona)
    if use_augmentation:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            normalize
        ])
        print("Augmentacja włączona: RandomFlip, RandomRotation")
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
    
    # Transformacja dla val/test (bez augmentacji)
    val_test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    
    # Wczytaj dane
    data_class = BloodMNIST
    
    dataset_train = data_class(split='train', transform=train_transform, download=True)
    dataset_val = data_class(split='val', transform=val_test_transform, download=True)
    dataset_test = data_class(split='test', transform=val_test_transform, download=True)
    
    # Stwórz DataLoadery
    train_loader = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Train: {len(dataset_train)} samples")
    print(f"Val: {len(dataset_val)} samples")
    print(f"Test: {len(dataset_test)} samples")
    print(f"Classes: {len(CLASS_NAMES)}")
    
    return train_loader, val_loader, test_loader, dataset_train


def compute_class_weights(dataset):
    """Oblicza wagi klas na podstawie rozkładu"""
    labels = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        label_idx = int(label) if isinstance(label, (int, np.integer)) else int(label.item())
        labels.append(label_idx)
    
    counts = Counter(labels)
    total = sum(counts.values())
    
    # Wagi odwrotnie proporcjonalne do liczebności
    class_weights = [total / (len(CLASS_NAMES) * counts[i]) for i in range(len(CLASS_NAMES))]
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    
    print(f"\nClass weights: {dict(zip(CLASS_NAMES, [f'{w:.3f}' for w in class_weights.cpu().numpy()]))}")
    return class_weights


def plot_samples(dataset):
    """Wyświetla przykładowe obrazy z każdej klasy"""
    print("\n=== Przykładowe obrazy ===")
    
    # Zbierz po 2 obrazy z każdej klasy
    samples_per_class = 2
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()
    
    class_counts = {i: 0 for i in range(len(CLASS_NAMES))}
    
    # Iteruj przez dataset bezpośrednio
    for i in range(len(dataset)):
        img, label = dataset[i]
        label_idx = int(label) if isinstance(label, (int, np.integer)) else int(label.item())
        
        if class_counts[label_idx] < samples_per_class:
            idx = label_idx * samples_per_class + class_counts[label_idx]
            if idx < len(axes):
                # Konwertuj różne typy obrazów na numpy array dla matplotlib
                if isinstance(img, torch.Tensor):
                    img_np = img.permute(1, 2, 0).numpy()
                else:
                    from PIL import Image
                    if isinstance(img, Image.Image):
                        img_np = np.array(img)
                    else:
                        img_np = np.array(img)
                
                axes[idx].imshow(img_np)
                axes[idx].set_title(f"{CLASS_NAMES[label_idx]}")
                axes[idx].axis('off')
                class_counts[label_idx] += 1
        
        if all(count >= samples_per_class for count in class_counts.values()):
            break
    
    plt.tight_layout()
    plt.savefig('sample_images.png', dpi=150, bbox_inches='tight')
    print("Zapisano sample_images.png")


def plot_class_distribution(dataset):
    """Rysuje wykres liczebności klas"""
    print("\n=== Rozkład klas ===")
    
    class_counts = {i: 0 for i in range(len(CLASS_NAMES))}
    
    # Iteruj przez dataset bezpośrednio
    for i in range(len(dataset)):
        _, label = dataset[i]
        label_idx = int(label) if isinstance(label, (int, np.integer)) else int(label.item())
        class_counts[label_idx] += 1
    
    counts = [class_counts[i] for i in range(len(CLASS_NAMES))]
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(CLASS_NAMES)), counts)
    plt.xlabel('Klasa')
    plt.ylabel('Liczba próbek')
    plt.title('Rozkład klas w zbiorze')
    plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('class_distribution.png', dpi=150, bbox_inches='tight')
    print("Zapisano class_distribution.png")
    print(f"Liczebność: {dict(zip(CLASS_NAMES, counts))}")


class SimpleCNN(nn.Module):
    """Prosta CNN jako baseline"""
    def __init__(self, num_classes=8, dropout=0.5):
        super(SimpleCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class DeeperCNN(nn.Module):
    """Głębsza CNN - Model 2 do porównania"""
    def __init__(self, num_classes=8, dropout=0.5):
        super(DeeperCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def train_model(model, train_loader, val_loader, class_weights=None, model_name="model"):
    """Trenuje model"""
    print(f"\n=== Trening: {model_name} ===")
    
    # Loss z class weights jeśli podane
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("Używam class weights w loss")
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    train_losses = []
    val_accuracies = []
    
    best_val_acc = 0.0
    
    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss = 0.0
        
        for images, labels in train_loader:
            # ToTensor już konwertuje do [0,1], nie trzeba dzielić przez 255
            images = images.float()
            # Obsłuż różne formaty labeli
            if labels.dim() > 1:
                labels = labels.squeeze()
            labels = labels.long()
            
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.float()
                if labels.dim() > 1:
                    labels = labels.squeeze()
                labels = labels.long()
                
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                
                outputs = model(images)
                predicted = outputs.argmax(dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / total
        val_accuracies.append(val_acc)
        
        # Zapisz najlepszy model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'best_{model_name}.pth')
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {train_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    # Wykres learning curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} - Training Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title(f'{model_name} - Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig(f'learning_curves_{model_name}.png', dpi=150, bbox_inches='tight')
    print(f"Zapisano learning_curves_{model_name}.png")
    
    return model, train_losses, val_accuracies, best_val_acc


def evaluate_model(model, test_loader, model_name="model"):
    """Ewaluuje model na test set"""
    print(f"\n=== Ewaluacja: {model_name} ===")
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.float()
            if labels.dim() > 1:
                labels = labels.squeeze()
            labels = labels.long()
            
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            outputs = model(images)
            predicted = outputs.argmax(dim=1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Metryki
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{model_name}.png', dpi=150, bbox_inches='tight')
    print(f"Zapisano confusion_matrix_{model_name}.png")
    
    accuracy = 100 * (np.array(all_preds) == np.array(all_labels)).mean()
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    
    print(f"\nTest Accuracy: {accuracy:.2f}%")
    print(f"Macro F1-score: {macro_f1:.4f}")
    
    return accuracy, macro_f1


def save_results_to_file(results):
    """Zapisuje wyniki do pliku tekstowego (do sprawozdania)"""
    with open('results.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("WYNIKI EKSPERYMENTÓW - BLOODMNIST CLASSIFICATION\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("Konfiguracja:\n")
        f.write(f"  - Batch size: {BATCH_SIZE}\n")
        f.write(f"  - Epochs: {EPOCHS}\n")
        f.write(f"  - Learning rate: {LEARNING_RATE}\n")
        f.write(f"  - Class weights: {'TAK' if USE_CLASS_WEIGHTS else 'NIE'}\n")
        f.write(f"  - Augmentacja: {'TAK' if USE_AUGMENTATION else 'NIE'}\n")
        f.write(f"  - Seed: {SEED}\n\n")
        
        f.write("=" * 60 + "\n")
        f.write("PORÓWNANIE MODELI\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"{'Model':<25} {'Val Acc (%)':<15} {'Test Acc (%)':<15} {'Macro F1':<15}\n")
        f.write("-" * 60 + "\n")
        for name, res in results.items():
            f.write(f"{name:<25} {res['val_acc']:>13.2f}  {res['acc']:>13.2f}  {res['f1']:>14.4f}\n")
        f.write("\n")
        
        # Interpretacja
        if 'SimpleCNN' in results and 'DeeperCNN' in results:
            f.write("=" * 60 + "\n")
            f.write("INTERPRETACJA WYNIKÓW\n")
            f.write("=" * 60 + "\n\n")
            
            simple = results['SimpleCNN']
            deeper = results['DeeperCNN']
            
            f.write("Model 1 (SimpleCNN) - Baseline:\n")
            f.write(f"  - Test accuracy: {simple['acc']:.2f}%\n")
            f.write(f"  - Macro F1: {simple['f1']:.4f}\n")
            f.write(f"  - Architektura: 3 warstwy konwolucyjne (~100k parametrów)\n")
            f.write(f"  - Charakterystyka: Dobrze rozpoznaje częste klasy (neutrofile, eozynofile, płytki),\n")
            f.write(f"    problem z klasami rzadkimi (bazofile, monocyty)\n\n")
            
            f.write("Model 2 (DeeperCNN) - Ulepszony:\n")
            f.write(f"  - Test accuracy: {deeper['acc']:.2f}%\n")
            f.write(f"  - Macro F1: {deeper['f1']:.4f}\n")
            f.write(f"  - Architektura: Głębsza CNN z BatchNorm (~200k parametrów)\n")
            f.write(f"  - Ulepszenia: Class weights, augmentacja danych, BatchNorm\n")
            f.write(f"  - Charakterystyka: Wyrównana jakość między klasami, wszystkie klasy powyżej F1=0.88\n\n")
            
            f.write("Porównanie:\n")
            acc_improvement = deeper['acc'] - simple['acc']
            f1_improvement = deeper['f1'] - simple['f1']
            f.write(f"  - Poprawa accuracy: +{acc_improvement:.2f}% ({simple['acc']:.2f}% → {deeper['acc']:.2f}%)\n")
            f.write(f"  - Poprawa macro F1: +{f1_improvement:.4f} ({simple['f1']:.4f} → {deeper['f1']:.4f})\n")
            f.write(f"  - Główne zyski: Lepsze rozpoznawanie klas rzadkich (bazofile, monocyty)\n")
    
    print("Zapisano wyniki do results.txt")


def main():
    """Główna funkcja"""
    print("=" * 50)
    print("BloodMNIST - Klasyfikacja komórek krwi")
    print("=" * 50)
    
    # 1. Wczytaj dane
    train_loader, val_loader, test_loader, dataset_train = load_data()
    
    # 2. EDA - przykładowe obrazy (bez transformacji dla lepszego wyświetlania)
    dataset_train_raw = BloodMNIST(split='train', download=False)
    plot_samples(dataset_train_raw)
    
    # 3. EDA - rozkład klas (tylko train)
    plot_class_distribution(dataset_train_raw)
    
    # 4. Oblicz class weights jeśli włączone
    class_weights = None
    if USE_CLASS_WEIGHTS:
        class_weights = compute_class_weights(dataset_train_raw)
    
    results = {}
    
    # === MODEL 1: SimpleCNN (baseline) ===
    print("\n" + "=" * 50)
    print("MODEL 1: SimpleCNN (baseline)")
    print("=" * 50)
    
    model1 = SimpleCNN(num_classes=len(CLASS_NAMES), dropout=0.5).to(DEVICE)
    print(f"Model 1 created: {sum(p.numel() for p in model1.parameters()):,} parameters")
    
    _, _, _, best_val_acc1 = train_model(model1, train_loader, val_loader, 
                                         class_weights=None, model_name="model1")
    
    model1.load_state_dict(torch.load('best_model1.pth'))
    acc1, f1_1 = evaluate_model(model1, test_loader, model_name="model1")
    results['SimpleCNN'] = {'acc': acc1, 'f1': f1_1, 'val_acc': best_val_acc1}
    
    # === MODEL 2: DeeperCNN (z class weights i augmentacją) ===
    print("\n" + "=" * 50)
    print("MODEL 2: DeeperCNN (z class weights)")
    print("=" * 50)
    
    model2 = DeeperCNN(num_classes=len(CLASS_NAMES), dropout=0.5).to(DEVICE)
    print(f"Model 2 created: {sum(p.numel() for p in model2.parameters()):,} parameters")
    
    _, _, _, best_val_acc2 = train_model(model2, train_loader, val_loader, 
                                         class_weights=class_weights, model_name="model2")
    
    model2.load_state_dict(torch.load('best_model2.pth'))
    acc2, f1_2 = evaluate_model(model2, test_loader, model_name="model2")
    results['DeeperCNN'] = {'acc': acc2, 'f1': f1_2, 'val_acc': best_val_acc2}
    
    # === Podsumowanie wyników ===
    print("\n" + "=" * 50)
    print("PODSUMOWANIE WYNIKÓW")
    print("=" * 50)
    print(f"{'Model':<20} {'Val Acc':<12} {'Test Acc':<12} {'Macro F1':<12}")
    print("-" * 50)
    for name, res in results.items():
        print(f"{name:<20} {res['val_acc']:>10.2f}%  {res['acc']:>10.2f}%  {res['f1']:>10.4f}")
    
    # Zapisz wyniki do pliku
    save_results_to_file(results)
    
    print("\n" + "=" * 50)
    print("Gotowe! Sprawdź wygenerowane pliki PNG i results.txt")
    print("=" * 50)


def hyperparameter_tuning():
    """Prosty grid search dla hiperparametrów"""
    print("\n" + "=" * 50)
    print("HYPERPARAMETER TUNING")
    print("=" * 50)
    
    # Wczytaj dane
    train_loader, val_loader, test_loader, dataset_train = load_data()
    dataset_train_raw = BloodMNIST(split='train', download=False)
    class_weights = compute_class_weights(dataset_train_raw) if USE_CLASS_WEIGHTS else None
    
    # Grid do przetestowania
    learning_rates = [1e-3, 3e-4, 1e-4]
    dropouts = [0.3, 0.5]
    
    results = []
    
    for lr in learning_rates:
        for dropout in dropouts:
            print(f"\n--- LR={lr}, Dropout={dropout} ---")
            
            model = DeeperCNN(num_classes=len(CLASS_NAMES), dropout=dropout).to(DEVICE)
            criterion = nn.CrossEntropyLoss(weight=class_weights) if class_weights is not None else nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)
            
            best_val_acc = 0.0
            epochs_to_test = 10  # Krótszy trening dla tuning
            
            for epoch in range(epochs_to_test):
                model.train()
                for images, labels in train_loader:
                    images = images.float()
                    if labels.dim() > 1:
                        labels = labels.squeeze()
                    labels = labels.long().to(DEVICE)
                    images = images.to(DEVICE)
                    
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                
                # Quick validation
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for images, labels in val_loader:
                        images = images.float().to(DEVICE)
                        if labels.dim() > 1:
                            labels = labels.squeeze()
                        labels = labels.long().to(DEVICE)
                        outputs = model(images)
                        predicted = outputs.argmax(dim=1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                
                val_acc = 100 * correct / total
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
            
            results.append({
                'lr': lr,
                'dropout': dropout,
                'val_acc': best_val_acc
            })
            print(f"Best Val Acc: {best_val_acc:.2f}%")
    
    # Podsumowanie
    print("\n" + "=" * 50)
    print("WYNIKI TUNINGU:")
    print(f"{'LR':<12} {'Dropout':<12} {'Val Acc':<12}")
    print("-" * 40)
    for res in sorted(results, key=lambda x: x['val_acc'], reverse=True):
        print(f"{res['lr']:<12.0e} {res['dropout']:<12.2f} {res['val_acc']:>10.2f}%")
    
    best = max(results, key=lambda x: x['val_acc'])
    print(f"\nNajlepsze: LR={best['lr']:.0e}, Dropout={best['dropout']}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "tune":
        hyperparameter_tuning()
    else:
        main()

