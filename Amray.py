import streamlit as st
from PIL import Image, ImageDraw
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection

st.set_page_config(page_title="Fracture Detection", layout="wide")
st.title("Détection de fracture - IA + Annotation manuelle")

# 🔹 Charger le token Hugging Face depuis les secrets Streamlit
HF_API_KEY = st.secrets["HUGGINGFACE"]["HF_API_KEY"]

# 🔹 Charger le modèle DETR pré-entraîné (fracture detection)
@st.cache_resource(show_spinner=False)
def load_model():
    processor = DetrImageProcessor.from_pretrained(
        "facebook/detr-resnet-50",
        use_auth_token=HF_API_KEY
    )
    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        use_auth_token=HF_API_KEY
    )
    return processor, model

processor, model = load_model()

# 🔹 Upload de l'image
uploaded_file = st.file_uploader("Choisissez une radiographie", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Radiographie chargée", use_column_width=True)
    
    # 🔹 Détection IA
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=0.5
    )[0]
    
    draw = ImageDraw.Draw(image)
    
    if len(results["scores"]) > 0:
        st.subheader("Fracture détectée automatiquement :")
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            draw.rectangle(box, outline="red", width=3)
        st.image(image, caption="Fracture détectée (rectangle rouge)", use_column_width=True)
    else:
        st.warning("Aucune fracture détectée automatiquement. Vous pouvez annoter manuellement ci-dessous.")

    # 🔹 Annotation manuelle
    st.subheader("Annotation manuelle (dessinez un rectangle si nécessaire)")
    x1 = st.number_input("x1", value=0)
    y1 = st.number_input("y1", value=0)
    x2 = st.number_input("x2", value=image.width)
    y2 = st.number_input("y2", value=image.height)
    if st.button("Ajouter annotation"):
        draw.rectangle([x1, y1, x2, y2], outline="blue", width=3)
        st.image(image, caption="Image annotée manuellement (rectangle bleu)", use_column_width=True)
