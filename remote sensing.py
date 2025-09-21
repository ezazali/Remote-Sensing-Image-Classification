import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.transforms import v2 as transforms_v2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import os
import cv2
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

# GPU Setup and optimization
def setup_gpu():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"GPU Available: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Enable optimizations for better GPU utilization
        torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes
        torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
        
        # Set memory allocation strategy
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
        
        return device
    else:
        print("CUDA not available, using CPU")
        return torch.device('cpu')

device = setup_gpu()

  

class ConvBNLeaky(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, alpha=0.1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, 
                             padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(alpha, inplace=True)
    
    def forward(self, x):
        return self.leaky(self.bn(self.conv(x)))

class FireModule(nn.Module):
    def __init__(self, in_channels, squeeze_filters=16, expand1x1=64, expand3x3=64):
        super().__init__()
        self.squeeze = nn.Sequential(
            nn.Conv2d(in_channels, squeeze_filters, 1, bias=False),
            nn.BatchNorm2d(squeeze_filters),
            nn.ReLU(inplace=True)
        )
        
        self.expand1x1 = nn.Sequential(
            nn.Conv2d(squeeze_filters, expand1x1, 1, bias=False),
            nn.BatchNorm2d(expand1x1),
            nn.ReLU(inplace=True)
        )
        
        self.expand3x3 = nn.Sequential(
            nn.Conv2d(squeeze_filters, expand3x3, 3, padding=1, bias=False),
            nn.BatchNorm2d(expand3x3),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        squeeze = self.squeeze(x)
        e1 = self.expand1x1(squeeze)
        e3 = self.expand3x3(squeeze)
        return torch.cat([e1, e3], dim=1)

class DenseBlock(nn.Module):
    def __init__(self, in_channels, num_layers=3, growth_rate=16):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_in_channels = in_channels + i * growth_rate
            self.layers.append(nn.Sequential(
                nn.BatchNorm2d(layer_in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(layer_in_channels, growth_rate, 3, padding=1, bias=False)
            ))
    
    def forward(self, x):
        concat_feat = x
        for layer in self.layers:
            new_feat = layer(concat_feat)
            concat_feat = torch.cat([concat_feat, new_feat], dim=1)
        return concat_feat

# -------------------------
# Main Model
# -------------------------
class NovelMultiscaleCNN(nn.Module):
    def __init__(self, input_shape=(3, 224, 224), num_classes=12, dropout_rate=0.5):
        super().__init__()
        
    
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        
        self.second_conv = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # Feature map generation
        self.feature_conv = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
         
        self.way1_dense = DenseBlock(128, num_layers=3, growth_rate=16)
        self.way1_conv = ConvBNReLU(128 + 3*16, 32)  # 128 + 3*16 = 176
        self.way1_dropout = nn.Dropout(dropout_rate)
        
        # Way 2: Bottleneck + conv stack
        self.way2_bottleneck = ConvBNReLU(128, 64, 1, 0)
        self.way2_mid1 = ConvBNReLU(64, 128)
        self.way2_mid2 = ConvBNReLU(128, 256, 1, 0)
        
        # Way 3: Fire module
        self.way3_fire = FireModule(128, squeeze_filters=32, expand1x1=64, expand3x3=64)
        
        # Way 4: Depthwise separable conv
        self.way4_conv = ConvBNReLU(128, 64)
        self.way4_depthwise = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, groups=64, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.way4_pointwise = ConvBNReLU(64, 64, 1, 0)
        
        # Final conv after concatenation
        # Total channels: 32 + 256 + 128 + 64 = 480
        self.final_conv = ConvBNReLU(480, 32, 7, 3)
        
        # Classification head
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # Initial convolutions
        x = self.first_conv(x)
        x = self.second_conv(x)
        feature_map = self.feature_conv(x)
        
        # Multi-scale processing
        # Way 1
        way1 = self.way1_dense(feature_map)
        way1 = self.way1_conv(way1)
        way1 = self.way1_dropout(way1)
        
        # Way 2
        way2 = self.way2_bottleneck(feature_map)
        way2 = self.way2_mid1(way2)
        way2 = self.way2_mid2(way2)
        
        # Way 3
        way3 = self.way3_fire(feature_map)
        
        # Way 4
        way4 = self.way4_conv(feature_map)
        way4 = self.way4_depthwise(way4)
        way4 = self.way4_pointwise(way4)
        
        # Concatenate all ways
        concat = torch.cat([way1, way2, way3, way4], dim=1)
        
        # Final processing
        concat = self.final_conv(concat)
        out = self.gap(concat)
        out = self.classifier(out)
        
        return out

# -------------------------
# Dataset Class
# -------------------------
class NWPUDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            # Convert to PIL Image for torchvision transforms
            image = transforms.ToPILImage()(image)
            image = self.transform(image)
        else:
            image = torch.FloatTensor(image).permute(2, 0, 1) / 255.0
            
        return image, label

# -------------------------
# Data Loading and Preprocessing
# -------------------------
def load_nwpu_dataset(dataset_path, target_size=(224, 224)):
    class_names = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
    print(f"Detected {len(class_names)} classes.")
    images, labels = [], []

    for idx, class_name in enumerate(tqdm(class_names, desc="Loading classes")):
        class_path = os.path.join(dataset_path, class_name)
        file_list = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_name in tqdm(file_list, desc=f"Loading {class_name}", leave=False):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, target_size)
            images.append(img)
            labels.append(idx)

    X = np.array(images, dtype=np.uint8)  # Keep as uint8 to save memory
    y = np.array(labels, dtype=np.int64)
    return X, y, class_names

def create_data_transforms():
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def split_data(X, y, test_size=0.15, val_size=0.15):
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_ratio, random_state=42, stratify=y_temp)
    return X_train, X_val, X_test, y_train, y_val, y_test

# -------------------------
# Training and Evaluation
# -------------------------
def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(tqdm(dataloader, desc="Training")):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
        
        if scaler:  # Mixed precision training
            with torch.cuda.amp.autocast():
                output = model(data)
                loss = criterion(output, target)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
    
    return total_loss / len(dataloader), correct / total

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in tqdm(dataloader, desc="Validating"):
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            
            with torch.cuda.amp.autocast():
                output = model(data)
                loss = criterion(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    return total_loss / len(dataloader), correct / total

def plot_training_history(train_losses, val_losses, train_accs, val_accs):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    axes[0].plot(epochs, train_accs, label='Train', marker='o')
    axes[0].plot(epochs, val_accs, label='Validation', marker='s')
    axes[0].set_title('Accuracy')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(epochs, train_losses, label='Train', marker='o')
    axes[1].plot(epochs, val_losses, label='Validation', marker='s')
    axes[1].set_title('Loss')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=200, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=200, bbox_inches='tight')
    plt.show()

def evaluate_model(model, test_loader, class_names, device):
    print("\n===== EVALUATION =====")
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Evaluating"):
            data, target = data.to(device), target.to(device)
            
            with torch.cuda.amp.autocast():
                output = model(data)
            
            pred = output.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-score: {f1:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0))
    
    plot_confusion_matrix(y_true, y_pred, class_names)
    return acc, prec, rec, f1

# -------------------------
# Main Training Function
# -------------------------
def main():
    # Hyperparameters optimized for GPU utilization
    DATASET_PATH = r"c:\Users\aipmu\OneDrive\Desktop\NWPU_RESISC45-20210923T210241Z-001\NWPU_RESISC45"
    BATCH_SIZE = 64  # Increased for better GPU utilization
    EPOCHS = 100
    INPUT_SHAPE = (3, 224, 224)
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    
    # Enable mixed precision training for better GPU utilization
    USE_MIXED_PRECISION = True
    
    print("🚀 NWPU-RESISC45 Training with PyTorch (GPU Optimized)")
    print(f"Using device: {device}")
    
    # Load and preprocess data
    print("\n📁 Loading dataset...")
    X, y, class_names = load_nwpu_dataset(DATASET_PATH, target_size=INPUT_SHAPE[1:])
    print(f"Total classes detected: {len(class_names)}")
    print(f"Dataset shape: {X.shape}")
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Create transforms
    train_transform, val_transform = create_data_transforms()
    
    # Create datasets
    train_dataset = NWPUDataset(X_train, y_train, train_transform)
    val_dataset = NWPUDataset(X_val, y_val, val_transform)
    test_dataset = NWPUDataset(X_test, y_test, val_transform)
    
    # Create data loaders with optimal settings for GPU
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False

    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False

    )
    
    # Create model
    print("\n🏗️ Building model...")
    model = NovelMultiscaleCNN(input_shape=INPUT_SHAPE, num_classes=len(class_names))
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing for better generalization
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if USE_MIXED_PRECISION and device.type == 'cuda' else None
    
    # Early stopping
    best_val_acc = 0
    patience = 15
    patience_counter = 0
    
    # Training history
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    print("\n🏋️ Starting training...")
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        
        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Save history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Early stopping and model saving
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"✅ New best model saved! Val Acc: {val_acc:.4f}")
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement")
            break
    
    # Load best model for evaluation
    model.load_state_dict(torch.load('best_model.pth'))
    
    # Plot training history
    plot_training_history(train_losses, val_losses, train_accs, val_accs)
    
    # Evaluate on test set
    acc, prec, rec, f1 = evaluate_model(model, test_loader, class_names, device)
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'class_names': class_names,
        'model_config': {
            'input_shape': INPUT_SHAPE,
            'num_classes': len(class_names)
        }
    }, 'novel_multiscale_cnn_pytorch.pth')

    
    print("\n🎉 Training completed!")
    print(f"Model saved: novel_multiscale_cnn_pytorch.pth")
    print(f"Final Test -> Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")

if __name__ == "__main__":
    main()
