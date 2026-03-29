import streamlit as st
from PIL import Image, ImageDraw
import requests
from io import BytesIO
import os

st.title("Détecteur de fracture avec IA - Prototype")

# --- Téléversement image ---
uploaded_file = st.file_uploader("Téléversez votre radiographie", type=["png", "jpg", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Radiographie originale", use_column_width=True)

    # --- Détection automatique avec Hugging Face DETR ---
    st.subheader("Détection automatique")
    try:
        from transformers import DetrImageProcessor, DetrForObjectDetection
        import torch

        HF_API_KEY = st.secrets["HUGGINGFACE"]["HF_API_KEY"]

        processor = DetrImageProcessor.from_pretrained(
            "facebook/detr-resnet-50", use_auth_token=HF_API_KEY
        )
        model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50", use_auth_token=HF_API_KEY
        )

        # Préparation de l'image
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)

        # Post-traitement
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.7)[0]

        if len(results["scores"]) == 0:
            st.warning("Le modèle n'a pas détecté de fracture.")
        else:
            draw = ImageDraw.Draw(image)
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                x1, y1, x2, y2 = box.tolist()
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            st.image(image, caption="Fracture détectée par IA", use_column_width=True)

    except Exception as e:
        st.error(f"Erreur détection IA : {e}")
        st.info("Vous pouvez utiliser l'annotation manuelle ci-dessous.")

    # --- Annotation manuelle ---
    st.subheader("Annotation manuelle")
    st.write("Si la détection automatique échoue, indiquez la fracture :")
    x1 = st.number_input("x1", value=50)
    y1 = st.number_input("y1", value=50)
    x2 = st.number_input("x2", value=200)
    y2 = st.number_input("y2", value=200)

    if st.button("Enregistrer annotation"):
        draw = ImageDraw.Draw(image)
        draw.rectangle([x1, y1, x2, y2], outline="blue", width=3)
        st.image(image, caption="Fracture annotée manuellement", use_column_width=True)

        save_path = "annotations"
        os.makedirs(save_path, exist_ok=True)
        annotation_file = os.path.join(save_path, uploaded_file.name + ".txt")
        with open(annotation_file, "w") as f:
            f.write(f"{x1},{y1},{x2},{y2}\n")
        st.success(f"Annotation sauvegardée pour apprentissage : {annotation_file}")
      Add Streamlit fracture detection script
