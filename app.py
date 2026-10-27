
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from model import PneumoniaCNN

st.set_page_config(
    page_title="Chest X-Ray AI",
    page_icon="🩻",
    layout="centered"
)

MODEL_PATH = "models/pneumonia_cnn_weighted.pth"

@st.cache_resource
def load_model():
    model = PneumoniaCNN()
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    )
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

st.title("🩻 Chest X-Ray Analysis")
st.write("Upload a chest X-ray image to get an AI model prediction.")

st.warning(
    "Educational prototype only. This model is not a medical "
    "diagnostic tool and must not be used for clinical decisions."
)

uploaded_file = st.file_uploader(
    "Upload a chest X-ray",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded X-ray", width=350)

    if st.button("Analyze Image"):
        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()

        classes = ["Normal", "Pneumonia-associated"]
        predicted_class = classes[prediction]
        confidence = probabilities[0][prediction].item() * 100

        st.subheader("Model Output")
        st.write(f"**Prediction:** {predicted_class}")
        st.write(f"**Model score:** {confidence:.2f}%")
