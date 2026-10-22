# Medical Image AI — PneumoniaMNIST Prototype

An educational machine-learning project that experiments with classifying small chest X-ray images as **Normal** or **Pneumonia** using the PneumoniaMNIST dataset.

> **Important:** This is a learning prototype, not a medical device. It must not be used to diagnose, rule out, or make treatment decisions about any person. Model outputs are not clinical probabilities.

## Project structure

```text
medical-image-ai/
├── app.py
├── model.py
├── requirements.txt
├── notebooks/
│   └── train_model.ipynb
└── models/
    └── .gitkeep
```

## 1. Train the model in Google Colab

1. Open `notebooks/train_model.ipynb` in Google Colab.
2. Run the cells from top to bottom.
3. The notebook trains a small CNN and saves `pneumonia_cnn.pth`.
4. Download the saved file and put it in the project's `models/` folder.

The notebook uses the public PneumoniaMNIST dataset distributed by MedMNIST. Images are 28×28 pixels, so this project is intended for learning and experimentation only.

## 2. Run the Streamlit app locally

Python 3.11 is recommended.

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

After placing the trained weights at `models/pneumonia_cnn.pth`, run:

```bash
streamlit run app.py
```

If the model file is missing, the app will explain how to create it rather than attempting to make a prediction.

## Dataset and references

- [MedMNIST](https://medmnist.com/)
- [PneumoniaMNIST documentation](https://medmnist.com/)
- [Google Colab](https://colab.research.google.com/)
- [Streamlit documentation](https://docs.streamlit.io/)

## Planned improvements

- Track validation loss and classification metrics.
- Review the confusion matrix and per-class recall.
- Add an explainability experiment such as Grad-CAM.
- Add tests and document limitations and dataset provenance.
