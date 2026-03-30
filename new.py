import streamlit as st
from PIL import Image, ImageDraw
import torch
import torchvision
from torchvision import transforms as T
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Détection de fracture", layout="wide")
st.title("Détection de fracture (IA + Annotation interactive)")

# Charger modèle FasterRCNN
@st.cache_resource(show_spinner=False)
def load_model():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model

model = load_model()

def transform_image(image):
    transform = T.Compose([T.ToTensor()])
    return transform(image)

uploaded_file = st.file_uploader("Choisissez une radiographie", type=["png", "jpg", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    # Détection automatique
    img_tensor = transform_image(image)
    with torch.no_grad():
        predictions = model([img_tensor])[0]

    draw = ImageDraw.Draw(image)
    detected = False
    threshold = 0.7
    for box, score in zip(predictions['boxes'], predictions['scores']):
        if score >= threshold:
            box = [round(i.item()) for i in box]
            draw.rectangle(box, outline="red", width=3)
            detected = True

    if detected:
        st.subheader("Fracture détectée automatiquement (rectangle rouge)")
    else:
        st.warning("Aucune fracture détectée automatiquement. Vous pouvez annoter manuellement.")

    # Affiche l'image sur le canvas pour annotation
    st.subheader("Annotation manuelle (dessinez directement sur l'image)")
    canvas_result = st_canvas(
        fill_color="rgba(0,0,255,0.3)",  # couleur bleu transparente
        stroke_width=3,
        stroke_color="blue",
        background_image=image,
        update_streamlit=True,
        height=image.height,
        width=image.width,
        drawing_mode="rect",
        key="canvas"
    )

    # Afficher l'image annotée
    if canvas_result.image_data is not None:
        annotated_image = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGB')
        st.image(annotated_image, caption="Image annotée", use_column_width=True)
