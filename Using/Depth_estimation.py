
from pathlib import Path

import cv2
import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
from torchvision.models import mobilenet_v2


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "ANCHURUS(v.5).pth"

IMG_SIZE = (256, 256)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ANCHURUS(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()

        base_model = mobilenet_v2(pretrained=pretrained)
        self.encoder = base_model.features

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
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Upsample(
                scale_factor=2,
                mode="bilinear",
                align_corners=False
            )
        )

    def forward(self, x):
        features = self.encoder(x)
        return self.decoder(features)


def load_model(checkpoint_path=MODEL_PATH, device=DEVICE):
    model = ANCHURUS(pretrained=False).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(checkpoint)
    model.eval()

    return model


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def run_webcam():
    model = load_model()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    print("Starting depth estimation... Press 'q' to quit.")

    try:
        with torch.no_grad():
            while cap.isOpened():
                ret, frame = cap.read()

                if not ret:
                    break

                input_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                input_tensor = transform(input_frame).unsqueeze(0).to(DEVICE)

                depth_output = model(input_tensor)

                depth_map = depth_output.squeeze().cpu().numpy()
                depth_map = (depth_map * 255).astype(np.uint8)

                depth_color = cv2.applyColorMap(
                    depth_map,
                    cv2.COLORMAP_MAGMA
                )

                depth_resized = cv2.resize(
                    depth_color,
                    (frame.shape[1], frame.shape[0])
                )

                combined = np.hstack((frame, depth_resized))

                cv2.imshow(
                    "RGB (Left) / Depth (Right)",
                    combined
                )

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_webcam()
