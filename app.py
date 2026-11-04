
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from model import PneumoniaCNN


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Chest X-Ray AI",
    page_icon="🩻",
    layout="centered"
)


# --------------------------------------------------
# MODEL CONFIGURATION
# --------------------------------------------------

MODEL_PATH = "models/pneumonia_cnn_weighted.pth"


@st.cache_resource
def load_model():
    model = PneumoniaCNN()

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location="cpu",
            weights_only=True
        )
    )

    model.eval()

    return model


model = load_model()


# --------------------------------------------------
# IMAGE TRANSFORMATION
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])


# --------------------------------------------------
# APPLICATION UI
# --------------------------------------------------

st.title("🩻 Chest X-Ray Analysis")

st.write(
    "Upload a chest X-ray image to get an AI model prediction."
)

st.warning(
    "Educational prototype only. This model is not a medical "
    "diagnostic tool and must not be used for clinical decisions."
)


# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a chest X-ray",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded X-ray",
        width=350
    )

    if st.button("Analyze Image", type="primary"):

        input_tensor = transform(image).unsqueeze(0)

        # MODEL PREDICTION
        with torch.no_grad():

            output = model(input_tensor)

            probabilities = torch.softmax(output, dim=1)

            prediction = torch.argmax(
                probabilities,
                dim=1
            ).item()

        # CLASS SCORES
        classes = [
            "Normal",
            "Pneumonia-associated"
        ]

        normal_score = probabilities[0][0].item() * 100

        pneumonia_score = probabilities[0][1].item() * 100

        predicted_class = classes[prediction]


        # --------------------------------------------------
        # RESULTS
        # --------------------------------------------------

        st.divider()

        st.subheader("Analysis Results")

        if prediction == 0:

            st.success("Model Prediction: Normal")

        else:

            st.warning(
                "Model Prediction: Pneumonia-associated"
            )


        # --------------------------------------------------
        # MODEL SCORES
        # --------------------------------------------------

        st.write("### Model Scores")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                label="Normal",
                value=f"{normal_score:.2f}%"
            )

        with col2:

            st.metric(
                label="Pneumonia-associated",
                value=f"{pneumonia_score:.2f}%"
            )


        # --------------------------------------------------
        # SCORE DISTRIBUTION
        # --------------------------------------------------

        st.write("### Score Distribution")

        st.write("Normal")

        st.progress(normal_score / 100)

        st.caption(
            f"Model score: {normal_score:.2f}%"
        )

        st.write("Pneumonia-associated")

        st.progress(pneumonia_score / 100)

        st.caption(
            f"Model score: {pneumonia_score:.2f}%"
        )

        st.divider()

        st.caption(
            "These scores represent the model's softmax outputs. "
            "They are not clinically calibrated probabilities "
            "and should not be used for medical decisions."
        )
