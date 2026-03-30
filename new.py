import streamlit as st
from PIL import Image, ImageDraw
import torch
import torchvision
from torchvision import transforms as T

st.set_page_config(page_title="Détection de fracture", layout="wide")
st.title("Détection de fracture (IA + Annotation)")

# Charger modèle FasterRCNN pré-entraîné
@st.cache_resource(show_spinner=False)
def load_model():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model

model = load_model()

# Transformation image
def transform_image(image):
    transform = T.Compose([T.ToTensor()])
    return transform(image)

# Upload image
uploaded_file = st.file_uploader("Choisissez une radiographie", type=["png", "jpg", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Radiographie chargée", use_column_width=True)

    # Détection
    img_tensor = transform_image(image)
    with torch.no_grad():
        predictions = model([img_tensor])[0]

    draw = ImageDraw.Draw(image)
    detected = False
    threshold = 0.7  # Seuil confiance
    for box, score in zip(predictions['boxes'], predictions['scores']):
        if score >= threshold:
            box = [round(i.item()) for i in box]
            draw.rectangle(box, outline="red", width=3)
            detected = True

    if detected:
        st.subheader("Fracture détectée automatiquement (rectangle rouge)")
    else:
        st.warning("Aucune fracture détectée automatiquement. Vous pouvez annoter manuellement ci-dessous.")

    st.image(image, use_column_width=True)

    # Annotation manuelle
    st.subheader("Annotation manuelle (rectangle bleu)")
    x1 = st.number_input("x1", value=0)
    y1 = st.number_input("y1", value=0)
    x2 = st.number_input("x2", value=image.width)
    y2 = st.number_input("y2", value=image.height)
    if st.button("Ajouter annotation"):
        draw.rectangle([x1, y1, x2, y2], outline="blue", width=3)
        st.image(image, caption="Image annotée manuellement (rectangle bleu)", use_column_width=True)
