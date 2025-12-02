# BloodMNIST - Blood Cell Classification

Deep learning project for classifying 8 types of blood cells from microscopic images using the BloodMNIST dataset (MedMNIST collection).

## Dataset

The BloodMNIST dataset contains **17,092 microscopic images** of blood cells divided into **8 classes**:
- Basophil
- Eosinophil
- Erythroblast
- Immature Granulocyte
- Lymphocyte
- Monocyte
- Neutrophil
- Platelet

**Image specifications:**
- Resolution: **28×28 pixels** (RGB)
- Split: Train/Val/Test = **7:1:2**
  - Training set: **11,959 samples**
  - Validation set: **1,712 samples**
  - Test set: **3,421 samples**

The dataset is moderately imbalanced, with neutrophil (2,330) and eosinophil (2,181) being the most frequent classes, while basophil (852) and lymphocyte (849) are less common.

## Setup

### Create virtual environment:

```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

### Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Full pipeline (EDA + both models):

```bash
python bloodmnist.py
```

This will:
1. Download and load the BloodMNIST dataset
2. Perform exploratory data analysis (EDA)
3. Train Model 1: SimpleCNN (baseline)
4. Train Model 2: DeeperCNN (with class weights)
5. Evaluate both models on the test set
6. Generate visualizations and save results

### Hyperparameter tuning:

```bash
python bloodmnist.py tune
```

## Project Structure

### Models

**Model 1: SimpleCNN (Baseline)**
- Architecture: 3 convolutional layers → AdaptiveAvgPool → 2 fully connected layers
- Parameters: **102,024**
- Features: Basic CNN architecture, dropout (0.5)
- Results:
  - Test Accuracy: **83.25%**
  - Macro F1-score: **0.7870**
  - Characteristics: Good performance on frequent classes (neutrophil, eosinophil, platelet), struggles with rare classes (basophil, monocyte)

**Model 2: DeeperCNN (Improved)**
- Architecture: Deeper CNN with BatchNorm, 4 convolutional blocks → classifier with 3 FC layers
- Parameters: **313,192**
- Features: BatchNorm layers, class weights in loss function, data augmentation, dropout (0.5)
- Results:
  - Test Accuracy: **94.04%**
  - Macro F1-score: **0.9319**
  - Characteristics: Balanced performance across all classes, all classes achieve F1 > 0.88

**Improvement:** +10.79% accuracy and +0.1449 macro F1-score compared to baseline.

### Configuration

Key settings in `bloodmnist.py`:

- `USE_CLASS_WEIGHTS = True` - Class weights in loss function to handle class imbalance
- `USE_AUGMENTATION = True` - Data augmentation (RandomHorizontalFlip, RandomRotation)
- `LEARNING_RATE = 0.001` - Learning rate for Adam optimizer
- `BATCH_SIZE = 64` - Batch size for training
- `EPOCHS = 20` - Number of training epochs
- `SEED = 42` - Random seed for reproducibility

## Generated Files

### Visualizations

- **`sample_images.png`** - Sample images from each blood cell class (2 per class)
- **`class_distribution.png`** - Bar chart showing class frequency distribution
- **`learning_curves_model1.png`** - Training loss and validation accuracy curves for SimpleCNN
- **`learning_curves_model2.png`** - Training loss and validation accuracy curves for DeeperCNN
- **`confusion_matrix_model1.png`** - Confusion matrix heatmap for SimpleCNN
- **`confusion_matrix_model2.png`** - Confusion matrix heatmap for DeeperCNN

### Models & Results

- **`best_model1.pth`** - Trained SimpleCNN model (best validation accuracy)
- **`best_model2.pth`** - Trained DeeperCNN model (best validation accuracy)
- **`results.txt`** - Detailed results summary with configuration, metrics, and interpretation (ready for report)

## Results Summary

| Model | Validation Acc | Test Accuracy | Macro F1 | Parameters |
|-------|----------------|---------------|----------|------------|
| SimpleCNN | 83.76% | **83.25%** | **0.7870** | 102,024 |
| DeeperCNN | 94.80% | **94.04%** | **0.9319** | 313,192 |

**Key improvements:**
- +10.79% absolute accuracy improvement
- +0.1449 macro F1-score improvement
- Better performance on rare classes (basophil recall: 0.45 → 0.96, monocyte recall: 0.50 → 0.92)

## Technologies

- **Python** 3.11+
- **PyTorch** - Deep learning framework
- **MedMNIST** - Dataset collection
- **scikit-learn** - Metrics and evaluation
- **matplotlib**, **seaborn** - Visualizations
- **NumPy**, **Pandas** - Data processing

## Methodology

### Data Preprocessing
- Normalization: Mean=[0.5, 0.5, 0.5], Std=[0.5, 0.5, 0.5]
- Data augmentation (training only): Random horizontal flip (p=0.5), Random rotation (±15°)

### Training
- Optimizer: Adam with learning rate 0.001
- Loss function: CrossEntropyLoss (with class weights for Model 2)
- Early stopping: Best model saved based on validation accuracy
- Reproducibility: All random seeds set to 42

### Evaluation
- Metrics: Accuracy, Precision, Recall, F1-score (per class and macro average)
- Visualization: Confusion matrices, learning curves, classification reports

## License

MIT License - see LICENSE file for details
