import streamlit as st
import torch
import matplotlib.pyplot as plt
from generator import Generator, latent_dim

# Load model
device = "cpu"
G = Generator()
G.load_state_dict(torch.load("mnist_generator.pth", map_location=device))
G.eval()

st.title("🧠 MNIST Digit Generator")
digit = st.selectbox("Pick a digit (0–9)", list(range(10)))

if st.button("Generate Images"):
    z = torch.randn(5, latent_dim)
    labels = torch.full((5,), digit, dtype=torch.long)
    with torch.no_grad():
        imgs = G(z, labels).cpu()

    # Show images
    fig, axs = plt.subplots(1, 5, figsize=(10, 2))
    for i in range(5):
        axs[i].imshow(imgs[i].squeeze(), cmap="gray")
        axs[i].axis("off")
    st.pyplot(fig)
