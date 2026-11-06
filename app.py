
import json
import streamlit as st
import torch
import pandas as pd

from pathlib import Path
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
# PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "pneumonia_cnn_weighted.pth"
EVALUATION_PATH = BASE_DIR / "evaluation_results.json"


# --------------------------------------------------
# MODEL CONFIGURATION
# --------------------------------------------------

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
# LOAD EVALUATION RESULTS
# --------------------------------------------------

@st.cache_data
def load_evaluation_results():

    with open(EVALUATION_PATH, "r") as file:
        return json.load(file)


evaluation_results = load_evaluation_results()


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
        This project explores the use of deep learning
        for chest X-ray image classification.

        The model predicts whether an image belongs
        to the Normal or Pneumonia-associated class.
        """
    )

    st.divider()

    st.caption("Educational project only.")


# --------------------------------------------------
# MAIN DASHBOARD
# --------------------------------------------------

st.title("Chest X-Ray Analysis Dashboard")

st.write(
    "Explore chest X-ray classification and model performance."
)

st.warning(
    "Educational prototype only. This application is not a "
    "medical diagnostic tool and must not be used "
    "for clinical decisions."
)


# --------------------------------------------------
# DASHBOARD TABS
# --------------------------------------------------

analysis_tab, evaluation_tab = st.tabs([
    "🩻 Image Analysis",
    "📊 Model Evaluation"
])


# ==================================================
# TAB 1: IMAGE ANALYSIS
# ==================================================

with analysis_tab:

    st.subheader("Upload X-Ray Image")

    uploaded_file = st.file_uploader(
        "Choose a chest X-ray image",
        type=["jpg", "jpeg", "png"],
        key="xray_uploader"
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
            st.write(
                f"**Image size:** {image.size[0]} × {image.size[1]}"
            )
            st.write("**Format:** RGB")
            st.write("**Model input:** 28 × 28 Grayscale")

            st.divider()

            analyze_button = st.button(
                "🔍 Analyze Image",
                type="primary",
                use_container_width=True
            )

        if analyze_button:

            input_tensor = transform(image).unsqueeze(0)

            with torch.no_grad():

                output = model(input_tensor)

                probabilities = torch.softmax(output, dim=1)

                prediction = torch.argmax(
                    probabilities,
                    dim=1
                ).item()

            normal_score = probabilities[0][0].item() * 100
            pneumonia_score = probabilities[0][1].item() * 100

            st.divider()

            st.subheader("Analysis Results")

            if prediction == 0:

                st.success("Model Prediction: Normal")

            else:

                st.warning(
                    "Model Prediction: Pneumonia-associated"
                )

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

            st.write("### Score Distribution")

            st.write("Normal")
            st.progress(normal_score / 100)

            st.write("Pneumonia-associated")
            st.progress(pneumonia_score / 100)

            st.divider()

            st.caption(
                "These values are model softmax scores, not "
                "clinically calibrated probabilities."
            )

    else:

        st.info(
            "Please upload a chest X-ray image to start the analysis."
        )


# ==================================================
# TAB 2: MODEL EVALUATION
# ==================================================

with evaluation_tab:

    st.subheader("Model Performance")

    st.write(
        "Evaluation metrics calculated on the test dataset."
    )

    # ACCURACY

    accuracy = evaluation_results["accuracy"] * 100

    normal = evaluation_results["normal"]
    pneumonia = evaluation_results["pneumonia"]

    # SUMMARY METRICS

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Accuracy",
            f"{accuracy:.2f}%"
        )

    with col2:

        st.metric(
            "Normal F1-score",
            f"{normal['f1_score'] * 100:.2f}%"
        )

    with col3:

        st.metric(
            "Pneumonia F1-score",
            f"{pneumonia['f1_score'] * 100:.2f}%"
        )

    with col4:

        st.metric(
            "Pneumonia Recall",
            f"{pneumonia['recall'] * 100:.2f}%"
        )

    st.divider()

    # CLASSIFICATION REPORT

    st.subheader("Classification Report")

    report_data = {
        "Metric": ["Precision", "Recall", "F1-score", "Support"],
        "Normal": [
            normal["precision"],
            normal["recall"],
            normal["f1_score"],
            normal["support"]
        ],
        "Pneumonia": [
            pneumonia["precision"],
            pneumonia["recall"],
            pneumonia["f1_score"],
            pneumonia["support"]
        ]
    }

    report_df = pd.DataFrame(report_data)

    st.dataframe(
        report_df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Precision": st.column_config.NumberColumn(
                "Precision",
                format="%.3f"
            ),
            "Recall": st.column_config.NumberColumn(
                "Recall",
                format="%.3f"
            ),
            "F1-score": st.column_config.NumberColumn(
                "F1-score",
                format="%.3f"
            )
        }
    )

    st.divider()

    # CONFUSION MATRIX

    st.subheader("Confusion Matrix")

    cm = evaluation_results["confusion_matrix"]

    confusion_df = pd.DataFrame(
        cm,
        index=["Actual Normal", "Actual Pneumonia"],
        columns=["Predicted Normal", "Predicted Pneumonia"]
    )

    st.dataframe(
        confusion_df,
        use_container_width=True
    )

    st.caption(
        "Rows represent actual classes; columns represent "
        "predicted classes."
    )

    st.divider()

    # METRIC COMPARISON

    st.subheader("Class Performance Comparison")

    comparison_data = pd.DataFrame({
        "Class": ["Normal", "Pneumonia"],
        "Precision": [
            normal["precision"],
            pneumonia["precision"]
        ],
        "Recall": [
            normal["recall"],
            pneumonia["recall"]
        ],
        "F1-score": [
            normal["f1_score"],
            pneumonia["f1_score"]
        ]
    })

    chart_data = comparison_data.set_index("Class")

    st.bar_chart(
        chart_data,
        stack=False,
        y_label="Score",
        height=350
    )

    st.divider()

    st.caption(
        "These metrics describe performance on the test dataset. "
        "They do not establish clinical effectiveness or "
        "real-world diagnostic accuracy."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Chest X-Ray AI | Educational Deep Learning Project"
)
