import cv2
import torch
import numpy as np
from torchvision import transforms
from torchvision.models import mobilenet_v2
import torch.nn as nn

# Configuration
MODEL_PATH = "models/ANCHURUS(v.5).pth"
IMG_SIZE = (256, 256)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ANCHURUS model definition - must match training architecture
class ANCHURUS(nn.Module):
    def __init__(self):
        super().__init__()
        # MobileNetV2 pre-trained encoder
        base_model = mobilenet_v2(pretrained=True)
        self.encoder = base_model.features
        
        # Progressive upsampling decoder
        self.decoder = nn.Sequential(
            self._up_block(1280, 512),
            self._up_block(512, 256),
            self._up_block(256, 128),
            self._up_block(128, 64),
            self._up_block(64, 32),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def _up_block(self, in_c, out_c):
        """Upsampling block with convolution, batch norm, and bilinear interpolation"""
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )

    def forward(self, x):
        features = self.encoder(x)
        return self.decoder(features)


# Load pre-trained model
model = ANCHURUS().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Image preprocessing pipeline - ImageNet normalization for MobileNetV2
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Initialize webcam
cap = cv2.VideoCapture(0)

print("Starting depth estimation... Press 'q' to quit.")

# Main inference loop
with torch.no_grad():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Convert frame to RGB and normalize
        input_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = transform(input_frame).unsqueeze(0).to(DEVICE)

        # Run depth estimation
        depth_output = model(input_tensor)
        
        # Convert depth output to 8-bit image for display
        depth_map = depth_output.squeeze().cpu().numpy()
        depth_map = (depth_map * 255).astype(np.uint8)
        
        # Apply colormap for better depth visualization
        depth_color = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)
        
        # Resize depth to match original frame dimensions
        depth_resized = cv2.resize(depth_color, (frame.shape[1], frame.shape[0]))

        # Combine RGB and depth side-by-side for display
        combined = np.hstack((frame, depth_resized))

        cv2.imshow('RGB (Left) / Depth (Right)', combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()