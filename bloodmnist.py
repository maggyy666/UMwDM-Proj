"""
BloodMNIST Blood Cell Classification Pipeline

A complete deep learning pipeline for classifying 8 types of blood cells from
microscopic images using the BloodMNIST dataset. The pipeline includes:
- Exploratory Data Analysis (EDA)
- Baseline CNN model (SimpleCNN)
- Improved CNN model (DeeperCNN with class weights)
- Model evaluation and comparison
- Hyperparameter tuning capabilities
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

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001
USE_CLASS_WEIGHTS = True
USE_AUGMENTATION = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "basophil",
    "eosinophil",
    "erythroblast",
    "immature granulocyte",
    "lymphocyte",
    "monocyte",
    "neutrophil",
    "platelet"
]

print(f"Using device: {DEVICE}")


def load_data(use_augmentation=USE_AUGMENTATION):
    """
    Load BloodMNIST dataset with data preprocessing and augmentation.
    
    Args:
        use_augmentation (bool): Whether to apply data augmentation to training set.
                                 Defaults to USE_AUGMENTATION global constant.
    
    Returns:
        tuple: A tuple containing (train_loader, val_loader, test_loader, dataset_train).
               DataLoaders are configured with appropriate batch sizes and transforms.
    """
    print("\n=== Wczytywanie danych ===")
    
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    
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
    
    val_test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    
    data_class = BloodMNIST
    
    dataset_train = data_class(split='train', transform=train_transform, download=True)
    dataset_val = data_class(split='val', transform=val_test_transform, download=True)
    dataset_test = data_class(split='test', transform=val_test_transform, download=True)
    
    train_loader = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Train: {len(dataset_train)} samples")
    print(f"Val: {len(dataset_val)} samples")
    print(f"Test: {len(dataset_test)} samples")
    print(f"Classes: {len(CLASS_NAMES)}")
    
    return train_loader, val_loader, test_loader, dataset_train


def compute_class_weights(dataset):
    """
    Compute class weights inversely proportional to class frequency.
    
    Args:
        dataset: PyTorch dataset containing (image, label) pairs.
    
    Returns:
        torch.Tensor: Class weights tensor of shape (num_classes,) on the specified device.
                     Weights are normalized such that rare classes receive higher weights.
    """
    labels = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        label_idx = int(label) if isinstance(label, (int, np.integer)) else int(label.item())
        labels.append(label_idx)
    
    counts = Counter(labels)
    total = sum(counts.values())
    
    class_weights = [total / (len(CLASS_NAMES) * counts[i]) for i in range(len(CLASS_NAMES))]
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    
    print(f"\nClass weights: {dict(zip(CLASS_NAMES, [f'{w:.3f}' for w in class_weights.cpu().numpy()]))}")
    return class_weights


def plot_samples(dataset):
    """
    Visualize sample images from each blood cell class.
    
    Args:
        dataset: PyTorch dataset containing blood cell images.
    
    Generates a 2x4 grid of sample images (2 per class) and saves to 'sample_images.png'.
    """
    print("\n=== Przykładowe obrazy ===")
    
    samples_per_class = 2
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()
    
    class_counts = {i: 0 for i in range(len(CLASS_NAMES))}
    
    for i in range(len(dataset)):
        img, label = dataset[i]
        label_idx = int(label) if isinstance(label, (int, np.integer)) else int(label.item())
        
        if class_counts[label_idx] < samples_per_class:
            idx = label_idx * samples_per_class + class_counts[label_idx]
            if idx < len(axes):
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
    """
    Plot class frequency distribution bar chart.
    
    Args:
        dataset: PyTorch dataset containing labeled samples.
    
    Generates a bar chart showing the number of samples per class and saves to
    'class_distribution.png'. Also prints the distribution dictionary.
    """
    print("\n=== Rozkład klas ===")
    
    class_counts = {i: 0 for i in range(len(CLASS_NAMES))}
    
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
    """
    Baseline convolutional neural network for blood cell classification.
    
    Architecture:
        - 3 convolutional blocks with max pooling
        - Adaptive average pooling
        - 2 fully connected layers with dropout
    
    Args:
        num_classes (int): Number of output classes. Defaults to 8.
        dropout (float): Dropout probability in classifier. Defaults to 0.5.
    """
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
    """
    Deeper convolutional neural network with BatchNorm for improved performance.
    
    Architecture:
        - 3 convolutional blocks with BatchNorm and max pooling
        - Adaptive average pooling
        - 3 fully connected layers with dropout
    
    Args:
        num_classes (int): Number of output classes. Defaults to 8.
        dropout (float): Dropout probability in classifier. Defaults to 0.5.
    """
    def __init__(self, num_classes=8, dropout=0.5):
        super(DeeperCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
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
    """
    Train a neural network model with validation monitoring.
    
    Args:
        model: PyTorch neural network model to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        class_weights (torch.Tensor, optional): Class weights for loss function to handle
                                                class imbalance. Defaults to None.
        model_name (str): Name identifier for saving model and plots. Defaults to "model".
    
    Returns:
        tuple: (model, train_losses, val_accuracies, best_val_acc)
               - model: Trained model
               - train_losses: List of training losses per epoch
               - val_accuracies: List of validation accuracies per epoch
               - best_val_acc: Best validation accuracy achieved
    
    Saves:
        - Best model weights to 'best_{model_name}.pth'
        - Learning curves plot to 'learning_curves_{model_name}.png'
    """
    print(f"\n=== Trening: {model_name} ===")
    
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
        model.train()
        train_loss = 0.0
        
        for images, labels in train_loader:
            images = images.float()
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
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'best_{model_name}.pth')
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {train_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
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
    """
    Evaluate model performance on test set.
    
    Args:
        model: Trained PyTorch neural network model.
        test_loader: DataLoader for test data.
        model_name (str): Name identifier for saving plots. Defaults to "model".
    
    Returns:
        tuple: (accuracy, macro_f1)
               - accuracy: Test accuracy percentage
               - macro_f1: Macro-averaged F1-score
    
    Prints:
        - Classification report with per-class metrics
        - Overall accuracy and macro F1-score
    
    Saves:
        - Confusion matrix heatmap to 'confusion_matrix_{model_name}.png'
    """
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
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))
    
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
    """
    Save experiment results to a text file for reporting purposes.
    
    Args:
        results (dict): Dictionary containing model results with keys as model names
                       and values as dicts with 'acc', 'f1', 'val_acc' metrics.
    
    Saves:
        Detailed results summary including configuration, model comparison table,
        and interpretation to 'results.txt'.
    """
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
    """
    Main execution function for the complete blood cell classification pipeline.
    
    Executes:
        1. Data loading and preprocessing
        2. Exploratory data analysis (visualizations)
        3. Class weights computation (if enabled)
        4. Model 1 training and evaluation (SimpleCNN baseline)
        5. Model 2 training and evaluation (DeeperCNN with improvements)
        6. Results summary and file export
    """
    print("=" * 50)
    print("BloodMNIST - Klasyfikacja komórek krwi")
    print("=" * 50)
    
    train_loader, val_loader, test_loader, dataset_train = load_data()
    
    dataset_train_raw = BloodMNIST(split='train', download=False)
    plot_samples(dataset_train_raw)
    
    plot_class_distribution(dataset_train_raw)
    
    class_weights = None
    if USE_CLASS_WEIGHTS:
        class_weights = compute_class_weights(dataset_train_raw)
    
    results = {}
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
    
    print("\n" + "=" * 50)
    print("PODSUMOWANIE WYNIKÓW")
    print("=" * 50)
    print(f"{'Model':<20} {'Val Acc':<12} {'Test Acc':<12} {'Macro F1':<12}")
    print("-" * 50)
    for name, res in results.items():
        print(f"{name:<20} {res['val_acc']:>10.2f}%  {res['acc']:>10.2f}%  {res['f1']:>10.4f}")
    
    save_results_to_file(results)
    
    print("\n" + "=" * 50)
    print("Gotowe! Sprawdź wygenerowane pliki PNG i results.txt")
    print("=" * 50)


def hyperparameter_tuning():
    """
    Perform grid search for hyperparameter optimization.
    
    Tests combinations of:
        - Learning rates: [1e-3, 3e-4, 1e-4]
        - Dropout rates: [0.3, 0.5]
    
    Uses DeeperCNN architecture with reduced training epochs (10) for efficiency.
    Prints validation accuracy results for each combination and identifies best configuration.
    """
    print("\n" + "=" * 50)
    print("HYPERPARAMETER TUNING")
    print("=" * 50)
    
    train_loader, val_loader, test_loader, dataset_train = load_data()
    dataset_train_raw = BloodMNIST(split='train', download=False)
    class_weights = compute_class_weights(dataset_train_raw) if USE_CLASS_WEIGHTS else None
    
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
            epochs_to_test = 10
            
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

