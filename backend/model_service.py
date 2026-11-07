import os
import json
import io
import base64
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess
import matplotlib

# Headless matplotlib backend
matplotlib.use("Agg")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

# Paths
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_pretrained_EfficientNetB0_model.h5")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(PROJECT_ROOT, "original_repo", "best_pretrained_EfficientNetB0_model.h5")

CLASS_NAMES_PATH = os.path.join(PROJECT_ROOT, "models", "class_names.json")
if not os.path.exists(CLASS_NAMES_PATH):
    CLASS_NAMES_PATH = os.path.join(PROJECT_ROOT, "original_repo", "class_names.json")

SAMPLES_DIR = os.path.join(PROJECT_ROOT, "samples")
if not os.path.exists(SAMPLES_DIR):
    SAMPLES_DIR = os.path.join(PROJECT_ROOT, "original_repo", "screenshots")

IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.50

# Disease metadata
DISEASE_METADATA = {
    "Calculus": {
        "displayName": "Dental Calculus (Tartar)",
        "icon": "🪨",
        "badgeColor": "#F59E0B",
        "accentColor": "#D97706",
        "severity": "Moderate",
        "description": "Hardened dental plaque caused by mineral precipitation from saliva. Forms above and below the gumline.",
        "clinicalNotes": "Requires professional ultrasonic scaling and root planing. Unaddressed calculus exacerbates periodontal inflammation and bone loss.",
        "urgency": "Routine dental cleaning recommended"
    },
    "Caries": {
        "displayName": "Dental Caries (Cavity)",
        "icon": "🦷",
        "badgeColor": "#EF4444",
        "accentColor": "#DC2626",
        "severity": "High",
        "description": "Microbial breakdown of calcified dental tissues (enamel and dentin) caused by acid-producing bacterial biofilm.",
        "clinicalNotes": "Early demineralization may be managed with topical fluorides; cavitated lesions necessitate operative debridement and restoration.",
        "urgency": "Prompt clinical evaluation advised"
    },
    "Gingivitis": {
        "displayName": "Gingivitis",
        "icon": "🩸",
        "badgeColor": "#EC4899",
        "accentColor": "#DB2777",
        "severity": "Mild to Moderate",
        "description": "Non-destructive inflammation of the gingiva tissue surrounding the teeth, often presenting with erythema, edema, and bleeding on probing.",
        "clinicalNotes": "Reversible with meticulous oral hygiene, plaque control, chlorhexidine rinses, and professional prophylaxis.",
        "urgency": "Hygiene improvement and dental checkup"
    },
    "Hypodontia": {
        "displayName": "Hypodontia",
        "icon": "🔲",
        "badgeColor": "#8B5CF6",
        "accentColor": "#7C3AED",
        "severity": "Congenital / Structural",
        "description": "Developmental absence of one to six permanent teeth (excluding third molars), affecting occlusion and arch space.",
        "clinicalNotes": "Multidisciplinary management involving orthodontics, space maintenance, and eventual prosthetic or implant reconstruction.",
        "urgency": "Orthodontic & prosthodontic consultation"
    },
    "Mouth_Ulcer": {
        "displayName": "Oral Ulcer (Aphthous Stomatitis)",
        "icon": "⚪",
        "badgeColor": "#F43F5E",
        "accentColor": "#E11D48",
        "severity": "Moderate Discomfort",
        "description": "Painful focal mucosal erosion or ulceration exposing the underlying lamina propria, frequently idiopathic or stress-induced.",
        "clinicalNotes": "Typically self-limiting within 10-14 days. Persistent lesions exceeding two weeks mandate biopsy to rule out malignancy.",
        "urgency": "Symptomatic relief; monitor healing duration"
    },
    "Tooth_Discoloration": {
        "displayName": "Tooth Discoloration",
        "icon": "🎨",
        "badgeColor": "#0D9488",
        "accentColor": "#0F766E",
        "severity": "Aesthetic / Variable",
        "description": "Extrinsic or intrinsic chromogenic staining caused by dietary agents, smoking, fluorosis, trauma, or aging.",
        "clinicalNotes": "Extrinsic stains resolve with polishing and prophy jet; intrinsic stains may require in-office bleaching, microabrasion, or veneers.",
        "urgency": "Aesthetic evaluation by dental clinician"
    }
}

# Benchmark metrics
MODEL_BENCHMARKS = {
    "architecture": "EfficientNetB0 + GAP + BatchNorm + Dropout(0.4) + Dense(6, softmax)",
    "trainingFramework": "TensorFlow / Keras",
    "weightsStatus": "Real pretrained checkpoint (best_pretrained_EfficientNetB0_model.h5)",
    "imageInputResolution": "224 x 224 x 3",
    "preprocessing": "EfficientNet standard scaling (via model's custom Lambda layer / numpy)",
    "metrics": {
        "accuracy": 0.8251,
        "precision": 0.8330,
        "recall": 0.8251,
        "f1Score": 0.8251
    },
    "comparativeModels": [
        {
            "model": "Custom CNN",
            "type": "Scratch Conv2D + GAP",
            "parameters": "~457K",
            "accuracy": 0.7578,
            "f1Score": 0.7518
        },
        {
            "model": "EfficientNetB0 (Deployed)",
            "type": "Transfer Learning (ImageNet fine-tuned)",
            "parameters": "~4.06M",
            "accuracy": 0.8251,
            "f1Score": 0.8251,
            "isBest": True
        },
        {
            "model": "ResNet50",
            "type": "Transfer Learning (ImageNet fine-tuned)",
            "parameters": "~23.6M",
            "accuracy": 0.7318,
            "f1Score": 0.7259
        }
    ]
}


class ModelService:
    def __init__(self):
        self.model = None
        self.grad_model = None
        self.class_names = []
        self._load()

    def _load(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model checkpoint not found at: {MODEL_PATH}")
        if not os.path.exists(CLASS_NAMES_PATH):
            raise FileNotFoundError(f"Class names file not found at: {CLASS_NAMES_PATH}")

        print(f"Loading EfficientNetB0 checkpoint from {MODEL_PATH}...")
        self.model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={"preprocess_input": effnet_preprocess}
        )

        with open(CLASS_NAMES_PATH, "r") as f:
            self.class_names = json.load(f)

        # Grad-CAM sub-model targeting final feature map
        try:
            target_layer = self.model.get_layer("top_activation")
            self.grad_model = tf.keras.models.Model(
                inputs=self.model.inputs,
                outputs=[target_layer.output, self.model.output]
            )
            print("Grad-CAM model initialized successfully targeting 'top_activation'.")
        except Exception as e:
            print(f"Notice: Could not construct Grad-CAM sub-model directly: {e}")
            self.grad_model = None

    def preprocess_image(self, pil_image: Image.Image) -> (np.ndarray, Image.Image):
        """Converts PIL image to 224x224 RGB float32 array."""
        rgb_image = pil_image.convert("RGB")
        resized_image = rgb_image.resize(IMG_SIZE, Image.Resampling.LANCZOS)
        arr = np.array(resized_image, dtype=np.float32)
        arr = np.expand_dims(arr, axis=0)
        return arr, resized_image

    def predict(self, pil_image: Image.Image) -> dict:
        arr, resized_image = self.preprocess_image(pil_image)
        raw_preds = self.model.predict(arr, verbose=0)[0]

        top_idx = int(np.argmax(raw_preds))
        top_class = self.class_names[top_idx]
        top_prob = float(raw_preds[top_idx])

        # Prediction distribution
        probabilities = []
        for idx, prob in enumerate(raw_preds):
            cls_name = self.class_names[idx]
            info = DISEASE_METADATA.get(cls_name, {})
            probabilities.append({
                "classKey": cls_name,
                "displayName": info.get("displayName", cls_name.replace("_", " ")),
                "icon": info.get("icon", "🔬"),
                "probability": float(prob),
                "badgeColor": info.get("badgeColor", "#3B82F6"),
                "severity": info.get("severity", "Unknown")
            })

        probabilities.sort(key=lambda x: x["probability"], reverse=True)

        is_uncertain = top_prob < CONFIDENCE_THRESHOLD
        primary_info = DISEASE_METADATA.get(top_class, {})

        return {
            "predictedClass": top_class,
            "displayName": primary_info.get("displayName", top_class.replace("_", " ")),
            "icon": primary_info.get("icon", "🔬"),
            "confidence": top_prob,
            "isUncertain": is_uncertain,
            "threshold": CONFIDENCE_THRESHOLD,
            "warningMessage": (
                "Confidence score is below the 50.0% threshold. The model does not include a healthy/normal class, "
                "meaning ambiguous or healthy scans are matched to the closest lesion. Please ensure optimal lighting and retake if necessary."
                if is_uncertain else None
            ),
            "diseaseInfo": primary_info,
            "probabilities": probabilities
        }

    def generate_gradcam(self, pil_image: Image.Image) -> dict:
        """Computes true gradient-weighted class activation mapping (Grad-CAM)."""
        if self.grad_model is None:
            return {"supported": False, "reason": "Grad-CAM target layer unavailable"}

        arr, resized_image = self.preprocess_image(pil_image)

        preds = self.model.predict(arr, verbose=0)[0]
        class_idx = int(np.argmax(preds))
        top_class = self.class_names[class_idx]
        top_prob = float(preds[class_idx])

        # Compute gradients for the predicted class
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(arr)
            loss = predictions[:, class_idx]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weight feature map channels by pooled gradients
        cam = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs[0]), axis=-1)
        cam = np.maximum(cam, 0)
        max_val = np.max(cam)
        if max_val > 0:
            cam = cam / max_val
        else:
            cam = np.zeros_like(cam)

        # Resize heatmap to input dimensions
        cam_pil = Image.fromarray((cam * 255).astype(np.uint8)).resize(IMG_SIZE, Image.Resampling.BICUBIC)
        cam_arr = np.array(cam_pil) / 255.0

        # Apply Jet colormap
        cmap = matplotlib.colormaps["jet"]
        jet_heatmap = cmap(cam_arr)[:, :, :3]
        jet_heatmap_uint8 = (jet_heatmap * 255).astype(np.uint8)
        jet_heatmap_pil = Image.fromarray(jet_heatmap_uint8)

        # Alpha-blend overlay onto original image
        orig_arr = np.array(resized_image, dtype=np.float32)
        overlay_arr = (orig_arr * 0.55 + jet_heatmap_uint8 * 0.45).clip(0, 255).astype(np.uint8)
        overlay_pil = Image.fromarray(overlay_arr)

        def to_b64(img: Image.Image) -> str:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

        return {
            "supported": True,
            "targetLayer": "top_activation (Conv2D: 7x7x1280)",
            "predictedClass": top_class,
            "confidence": top_prob,
            "originalBase64": to_b64(resized_image),
            "heatmapBase64": to_b64(jet_heatmap_pil),
            "overlayBase64": to_b64(overlay_pil)
        }

    def get_sample_images(self) -> list:
        samples = []
        if not os.path.exists(SAMPLES_DIR):
            return samples

        mapping = {
            "0.jpg": {"label": "Tooth Discoloration", "description": "Severe enamel chromogenic staining"},
            "images.jpg": {"label": "Dental Caries", "description": "Occlusal cavity and enamel breakdown"},
            "R.jpg": {"label": "Gingivitis", "description": "Marginal gingival erythema and inflammation"},
            "diskoloracija-zuba.jpg": {"label": "Dental Calculus", "description": "Supragingival tartar deposit"}
        }

        for filename, meta in mapping.items():
            fpath = os.path.join(SAMPLES_DIR, filename)
            if os.path.exists(fpath):
                try:
                    with Image.open(fpath) as img:
                        thumb = img.convert("RGB").resize((160, 160))
                        buf = io.BytesIO()
                        thumb.save(buf, format="JPEG", quality=85)
                        b64 = f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"
                        samples.append({
                            "id": filename,
                            "filename": filename,
                            "label": meta["label"],
                            "description": meta["description"],
                            "thumbnail": b64
                        })
                except Exception as e:
                    print(f"Error loading sample {filename}: {e}")

        return samples


# Global singleton instance
service = ModelService()
