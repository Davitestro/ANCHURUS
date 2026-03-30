import os
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision.models import mobilenet_v2
import cv2
import numpy as np
from torchvision import transforms
from tqdm import tqdm

# Configuration
BASE_DIR = "processed_data"
IMG_SIZE = (256, 256)
BATCH_SIZE = 32      
TRAIN_STEP_SIZE = 7000  # Virtual epoch size from 85k total frames
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Adaptive learning rate strategy parameters
BASE_LR = 1e-4         # Initial learning rate
KICK_LR = 1e-3         # High learning rate kick to escape training plateau
MIN_PROGRESS = 0.0001  # Accuracy improvement threshold to detect stagnation
PATIENCE = 4           # Epochs of slow progress before applying learning rate kick
MAX_KICKS = 3          # Maximum number of learning rate recovery attempts

# Dataset and Model Classes
class VideoDepthDataset(Dataset):
    """Load RGB images and corresponding depth maps from directory structure"""
    def __init__(self, base_dir):
        self.inputs, self.depths = [], []
        rgb_root, depth_root = os.path.join(base_dir, "rgb"), os.path.join(base_dir, "depth")
        # Collect all RGB-depth frame pairs from video folders
        for video_folder in os.listdir(rgb_root):
            r_path, d_path = os.path.join(rgb_root, video_folder), os.path.join(depth_root, video_folder)
            if os.path.isdir(r_path):
                frames = sorted(os.listdir(r_path))
                for f in frames:
                    self.inputs.append(os.path.join(r_path, f))
                    self.depths.append(os.path.join(d_path, f.replace(".jpg", ".png")))
        # ImageNet normalization for pre-trained MobileNetV2
        self.transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    def __len__(self): return len(self.inputs)
    def __getitem__(self, idx):
        # Load and resize RGB image and depth map
        img = cv2.resize(cv2.cvtColor(cv2.imread(self.inputs[idx]), cv2.COLOR_BGR2RGB), IMG_SIZE)
        depth = cv2.resize(cv2.imread(self.depths[idx], cv2.IMREAD_GRAYSCALE), IMG_SIZE)
        return self.transform(img), torch.tensor(depth/255., dtype=torch.float32).unsqueeze(0)

class DepthStudent(nn.Module):
    """MobileNetV2 encoder-decoder architecture for depth estimation"""
    def __init__(self):
        super().__init__()
        # Pre-trained encoder from ImageNet
        self.encoder = mobilenet_v2(pretrained=True).features
        # Progressive upsampling decoder
        self.decoder = nn.Sequential(
            self._up(1280, 512), self._up(512, 256), self._up(256, 128),
            self._up(128, 64), self._up(64, 32),
            nn.Conv2d(32, 1, 3, padding=1), nn.Sigmoid()
        )
    def _up(self, in_c, out_c):
        """Upsampling block with convolution, batch norm, and ReLU"""
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
    def forward(self, x): return self.decoder(self.encoder(x))

def get_acc(p, t):
    """Calculate delta-1 accuracy metric: % of pixels where max(pred/gt, gt/pred) < 1.25"""
    p, t = torch.clamp(p, 1e-3), torch.clamp(t, 1e-3)
    return (torch.max(p/t, t/p) < 1.25).float().mean().item()

# Prepare dataset and model
full_data = VideoDepthDataset(BASE_DIR)
train_raw, val_data = random_split(full_data, [int(0.9*len(full_data)), len(full_data)-int(0.9*len(full_data))])
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

model = DepthStudent().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR)
criterion = nn.MSELoss()

# Training state tracking for adaptive learning rate
best_acc = 0.0
stagnation_counter = 0
kicks_given = 0
is_in_kick_mode = False

print(f"🚀 Запуск реактивного обучения на {DEVICE}")
print(f"Базовый LR: {BASE_LR} | LR при пинке: {KICK_LR}")
print("-" * 50)

for epoch in range(300):
    # Randomly sample subset of training data for this epoch
    indices = np.random.choice(len(train_raw), TRAIN_STEP_SIZE, replace=False)
    train_loader = DataLoader(Subset(train_raw, indices), batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    
    model.train()
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    
    for imgs, depths in loop:
        imgs, depths = imgs.to(DEVICE), depths.to(DEVICE)
        preds = model(imgs)
        loss = criterion(preds, depths)
        
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        
        loop.set_postfix(
            loss=f"{loss.item():.4f}", 
            lr=f"{optimizer.param_groups[0]['lr']:.1e}"
        )

    # Evaluate on validation set
    model.eval()
    v_acc = 0
    with torch.no_grad():
        for imgs, depths in val_loader:
            imgs, depths = imgs.to(DEVICE), depths.to(DEVICE)
            v_acc += get_acc(model(imgs), depths)
    
    current_acc = v_acc / len(val_loader)
    progress = current_acc - best_acc
    
    print(f"\nEpoch {epoch+1} Results:")
    print(f"   Current Acc: {current_acc:.4f} | Best Acc: {best_acc:.4f} | Progress: {progress:.5f}")
    print(f"   Current LR:  {optimizer.param_groups[0]['lr']:.2e}")

    # Adaptive learning rate logic
    if progress > MIN_PROGRESS:
        # Good progress - reset stagnation counter and return to base LR if needed
        if current_acc > best_acc:
            best_acc = current_acc
            torch.save(model.state_dict(), "student_reactive_best.pth")
            print("   [NEW BEST] Model saved.")
        
        stagnation_counter = 0
        if is_in_kick_mode:
            # Return to normal learning rate after recovery
            for g in optimizer.param_groups: g['lr'] = BASE_LR
            is_in_kick_mode = False
            print("   [RECOVERED] Returning to base learning rate.")
    else:
        # Slow progress indicator - accumulate stagnation count
        stagnation_counter += 1
        print(f"   [STAGNATING] {stagnation_counter}/{PATIENCE}")

    # Apply learning rate kick to escape plateau if needed
    if stagnation_counter >= PATIENCE:
        if kicks_given < MAX_KICKS:
            kicks_given += 1
            stagnation_counter = 0
            is_in_kick_mode = True
            for g in optimizer.param_groups: g['lr'] = KICK_LR
            print(f"   [LR KICK] Applying recovery attempt #{kicks_given}")
        else:
            print(f"   [STOPPING] Max recovery attempts ({MAX_KICKS}) exhausted.")
            break

    print("-" * 50)