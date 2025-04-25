import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import cv2
from PIL import Image

# === Sidebar: Navigasi dan Informasi ===
st.sidebar.title("🧠 Batik Style Transfer")
st.sidebar.markdown("Aplikasi ini menggunakan model **CycleGAN** untuk mentransformasikan gambar flora menjadi motif batik Madura.")
st.sidebar.markdown("---")

st.sidebar.markdown("---")
st.sidebar.markdown("🔗 **Sampel Data Uji yang dapat digunakan**")
st.sidebar.markdown("[📁 Akses Data Uji](https://drive.google.com/drive/folders/15ATDaDFjeQDGBKxoh-w18X94b_6kX0C7?usp=sharing)", unsafe_allow_html=True)

# Pilih model pretrained
model_options = {
    "Style Pola Sederhana (Epoch 50)": "Pretrained/Pola Sederhana/G_epoch_50.pth",
    "Style Pola Kompleks (Epoch 50)": "Pretrained/Pola Kompleks/G_epoch_50.pth",
}
selected_model = st.sidebar.selectbox("📁 Pilih Model Pretrained", list(model_options.keys()))

# === Load Model ===
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        conv_block = [
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(in_features)
        ]
        self.conv_block = nn.Sequential(*conv_block)
    
    def forward(self, x):
        return x + self.conv_block(x)

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()
        model = [
            nn.Conv2d(input_nc, 64, kernel_size=7, stride=1, padding=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]
        in_features = 64
        out_features = in_features * 2

        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2

        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2

        model += [nn.Conv2d(64, output_nc, kernel_size=7, stride=1, padding=3), nn.Tanh()]
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)

def load_generator(model_path):
    G = Generator(input_nc=3, output_nc=3)
    G.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    G.eval()
    return G

G = load_generator(model_options[selected_model])

# === Image Transformations ===
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def process_image(image):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = G(image)
    output = output.squeeze(0).detach().numpy().transpose(1, 2, 0)
    output = (output * 0.5) + 0.5
    return np.clip(output, 0, 1)

# === Tampilan Utama ===
st.title("🎨 Transformasi Gaya Batik Madura")
st.markdown("""
Aplikasi ini bertujuan untuk mentransformasikan gambar flora menjadi motif batik khas Madura menggunakan **CycleGAN**.

1. Unggah satu atau lebih gambar flora.
2. Pilih model pretrained dari sidebar.
3. Lihat hasil transformasi dalam gaya batik!

""")

uploaded_files = st.file_uploader("📤 Upload Gambar (JPG/PNG)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    st.markdown("### 📸 Hasil Transformasi Gambar")

    cols = st.columns(3)

    for idx, uploaded_file in enumerate(uploaded_files):
        input_image = Image.open(uploaded_file).convert("RGB")
        input_image = input_image.resize((256, 256))
        output_image = process_image(input_image)
        output_image = (output_image * 255).astype(np.uint8)

        with cols[idx % 3]:
            # Gambar asli
            st.image(input_image, caption="Input", use_container_width=True)

            # Teks di tengah-tengah
            st.markdown(f"<div style='text-align: center; margin: 4px 0;'> {idx+1}</div>", unsafe_allow_html=True)

            # Gambar hasil
            st.image(output_image, caption="Transformasi", use_container_width=True)

        if (idx + 1) % 3 == 0 and (idx + 1) != len(uploaded_files):
            cols = st.columns(3)
