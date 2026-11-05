
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
    layout="wide"
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
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.title("🩻 Chest X-Ray AI")

    st.caption("Medical Image Analysis Project")

    st.divider()

    st.subheader("Project Information")

    st.write("**Project:** Pneumonia Classification")

    st.write("**Model:** Convolutional Neural Network")

    st.write("**Framework:** PyTorch")

    st.write("**Dataset:** PneumoniaMNIST")

    st.divider()

    st.subheader("Model Details")

    st.write("Input Size: 28 × 28")

    st.write("Input Channels: 1 (Grayscale)")

    st.write("Output Classes: 2")

    st.write("Architecture: CNN")

    st.divider()

    st.subheader("About")

    st.write(
        """
        This project explores the use of
        deep learning for chest X-ray image
        classification.

        The model predicts whether an image
        belongs to the Normal or
        Pneumonia-associated class.
        """
    )

    st.divider()

    st.caption("Developed as an educational project.")


# --------------------------------------------------
# MAIN DASHBOARD
# --------------------------------------------------

st.title("Chest X-Ray Analysis Dashboard")

st.write(
    "Upload a chest X-ray image and analyze it using "
    "our trained CNN model."
)

st.warning(
    "Educational prototype only. This application is not "
    "a medical diagnostic tool and must not be used "
    "for clinical decisions."
)

st.divider()


# --------------------------------------------------
# IMAGE UPLOAD SECTION
# --------------------------------------------------

st.subheader("📤 Upload X-Ray Image")

uploaded_file = st.file_uploader(
    "Choose a chest X-ray image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    image_col, info_col = st.columns([1, 1])

    with image_col:

        st.image(
            image,
            caption="Uploaded Chest X-Ray",
            use_container_width=True
        )

    with info_col:

        st.subheader("Image Information")

        st.write(f"**File name:** {uploaded_file.name}")

        st.write(f"**Image size:** {image.size[0]} × {image.size[1]}")

        st.write("**Format:** RGB")

        st.write("**Model input:** 28 × 28 Grayscale")

        st.divider()

        analyze_button = st.button(
            "🔍 Analyze Image",
            type="primary",
            use_container_width=True
        )


    # --------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------

    if analyze_button:

        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():

            output = model(input_tensor)

            probabilities = torch.softmax(output, dim=1)

            prediction = torch.argmax(
                probabilities,
                dim=1
            ).item()


        # CLASS SCORES

        normal_score = probabilities[0][0].item() * 100

        pneumonia_score = probabilities[0][1].item() * 100

        classes = [
            "Normal",
            "Pneumonia-associated"
        ]

        predicted_class = classes[prediction]


        # --------------------------------------------------
        # RESULTS SECTION
        # --------------------------------------------------

        st.divider()

        st.subheader("📊 Analysis Results")

        if prediction == 0:

            st.success("Model Prediction: Normal")

        else:

            st.warning(
                "Model Prediction: Pneumonia-associated"
            )


        # SCORE CARDS

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


        # SCORE DISTRIBUTION

        st.write("### Score Distribution")

        st.write("Normal")

        st.progress(normal_score / 100)

        st.write("Pneumonia-associated")

        st.progress(pneumonia_score / 100)


        st.divider()

        st.caption(
            "Scores represent the model's softmax outputs. "
            "They are not clinically calibrated probabilities "
            "and should not be used for medical decisions."
        )

else:

    st.info(
        "Please upload a chest X-ray image to start the analysis."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Chest X-Ray AI | Educational Deep Learning Project"
)
