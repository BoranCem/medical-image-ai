"""Streamlit demo for the educational PneumoniaMNIST classifier."""

from pathlib import Path

import streamlit as st
import torch
from PIL import Image, ImageOps
from torchvision import transforms

from model import PneumoniaCNN

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "pneumonia_cnn.pth"
CLASS_NAMES = ["Normal", "Pneumonia-associated"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(page_title="Medical Image AI", page_icon="🩻")
st.title("Chest X-ray Classification — Learning Demo")
st.warning(
    "Educational prototype only. This is not a medical device and must not be "
    "used for diagnosis, treatment, or other medical decisions."
)
st.write(
    "Upload an image to explore how a small model trained on PneumoniaMNIST "
    "behaves. The dataset uses 28×28 pixel images; results on other images may "
    "be unreliable."
)

if not MODEL_PATH.exists():
    st.info(
        "The trained model file was not found. Train the model using "
        "`notebooks/train_model.ipynb`, then place `pneumonia_cnn.pth` in "
        "`models/` and reload this page."
    )
    st.stop()


@st.cache_resource
def load_model():
    model = PneumoniaCNN().to(DEVICE)
    state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


uploaded_file = st.file_uploader(
    "Choose an image file", type=["png", "jpg", "jpeg", "bmp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_container_width=True)

    preprocess = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
        ]
    )
    tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(DEVICE)

    if st.button("Run model"):
        model = load_model()
        with torch.no_grad():
            logits = model(tensor)
            scores = torch.softmax(logits, dim=1)[0].cpu()

        predicted_index = int(torch.argmax(scores).item())
        st.subheader(f"Model output: {CLASS_NAMES[predicted_index]}")
        st.write("Relative model scores (not clinical probabilities):")
        st.bar_chart(
            {
                "Model score": {
                    CLASS_NAMES[0]: float(scores[0]),
                    CLASS_NAMES[1]: float(scores[1]),
                }
            }
        )
        st.caption(
            "These scores reflect this model's output only. They do not establish "
            "whether a person has pneumonia."
        )
