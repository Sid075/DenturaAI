# Dentura AI — Clinical Oral Pathology AI Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-19.0-61DAFB?logo=react&logoColor=black)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.x-3178C6?logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![Vite](https://img.shields.io/badge/Vite-8.x-646CFF?logo=vite&logoColor=white)](https://vitejs.dev/)

An end-to-end, clinical-grade oral pathology screening and diagnostic decision-support system. Powered by deep transfer learning on intraoral photographic imagery with pixel-level Grad-CAM spatial interpretability, multi-model comparative benchmarking, and an automated dataset integrity auditing suite.

---

## Key Highlights

- **Champion Classifier (EfficientNetB0)**: Fine-tuned with depthwise separable convolutions and squeeze-and-excitation optimization, achieving **82.51% test accuracy** and **82.52% weighted F1-score** across 1,115 held-out clinical test samples.
- **Pixel-Level Explainability (Grad-CAM)**: Backpropagates pathology-specific gradients into the final convolutional activation tensor (`top_activation`, shape: 7×7×1280) to render thermal heatmaps and alpha-blended overlays, localizing demineralization, calculus accretion, and mucosal ulcerations.
- **Dynamic Multi-Model Switching**: Seamless toggling between the champion **EfficientNetB0**, a lightweight **Custom 4-Block CNN** (~457K params), and a **ResNet50 Backbone** (~23.6M params) across both the Analysis Studio and the Evaluation Dashboard.
- **Model Evaluation Dashboard**: Real-time 6×6 interactive confusion matrix, per-class sensitivity/specificity/F1 metrics, confidence distributions, filterable misclassified image galleries, and automated CSV/JSON reporting.
- **Scientific Dataset Integrity Audit**: Automated cryptographic image hashing revealing cross-partition duplicate leakage and conflicting ground-truth labeling.

---

## Clinical Pathology Classes

Dentura AI detects and differentiates between 6 common dental and oral mucosal conditions:

| Pathology | Description | Clinical Action |
|---|---|---|
| **Dental Calculus** | Mineralized supragingival and subgingival plaque deposits | Routine scaling & root planing |
| **Dental Caries** | Bacterial biofilm demineralization of enamel and dentin | Operative restoration / remineralization |
| **Gingivitis** | Erythematous, non-destructive marginal gingival inflammation | Hygiene prophylaxis & plaque control |
| **Hypodontia** | Congenital absence of one or more permanent teeth | Orthodontic & prosthodontic alignment |
| **Oral Ulcer** | Focal mucosal erosions exposing underlying lamina propria | Symptomatic relief; malignancy rule-out |
| **Tooth Discoloration** | Extrinsic chromogenic or intrinsic developmental staining | Microabrasion, bleaching, or veneers |

---

## Architectural Comparison & Benchmarks

Empirical performance evaluation evaluated across 1,115 stratified test samples:

| Model Architecture | Model Type | Parameters | Accuracy | Balanced Acc | Macro F1 | Weighted F1 | Checkpoint Status |
|---|---|---|---|---|---|---|---|
| **EfficientNetB0 (Champion)** | Transfer Learning (ImageNet) | **~4.06M** | **82.51%** | **80.66%** | **82.78%** | **82.52%** | Trained (`models/best_pretrained_EfficientNetB0_model.h5`) |
| **Custom 4-Block CNN** | Scratch Conv2D + GAP | ~457K | 75.78% | 65.03% | 66.65% | 75.19% | Baseline Architecture |
| **ResNet50 Backbone** | Transfer Learning (ImageNet) | ~23.6M | 73.18% | 62.48% | 63.72% | 72.59% | Baseline Architecture |

---

## Repository Structure

```
Dentura_2/
├── backend/                       # High-performance asynchronous FastAPI service
│   ├── evaluation/                # Model evaluation suite & integrity audit engine
│   │   ├── evaluate_model.py      # Multi-model metrics calculator & confusion matrices
│   │   └── eval_cache.json        # Serialized evaluation cache
│   ├── main.py                    # REST API endpoints & route handlers
│   └── model_service.py           # TensorFlow model inference & Grad-CAM generator
├── frontend/                      # Modern React / TypeScript clinical interface
│   ├── src/
│   │   ├── components/
│   │   │   └── EvaluationDashboard.tsx  # Dynamic multi-model evaluation dashboard
│   │   ├── App.tsx                # Diagnostic Analysis Studio & landing page
│   │   └── index.css              # Custom clinical design system
│   └── package.json               # Node.js dependencies & scripts
├── models/                        # Serialized deep learning model checkpoints
│   ├── best_pretrained_EfficientNetB0_model.h5  # Champion weights
│   └── class_names.json           # Pathology class indices
├── notebooks/                     # Deep learning research & benchmarking notebooks
│   └── oral_pathology_training_and_benchmarks.ipynb
├── samples/                       # Curated intraoral diagnostic test scans
├── dataset/                       # Partitioned clinical image dataset
│   ├── train/                     # Training split
│   └── test/                      # Held-out evaluation split (1,115 samples)
├── requirements.txt               # Pinned Python package dependencies
└── README.md                      # Platform documentation
```

---

## Quickstart Guide

### 1. Backend Setup & Startup

Activate the Python virtual environment and run the FastAPI server:

```bash
# From project root
source venv/bin/activate
uvicorn main:app --app-dir backend --port 8008 --reload
```

Backend will be accessible at:
- API Base: `http://127.0.0.1:8008`
- Interactive OpenAPI Docs: `http://127.0.0.1:8008/docs`
- Health Check: `http://127.0.0.1:8008/health`

### 2. Frontend Setup & Startup

In a separate terminal, launch the Vite development server:

```bash
cd frontend
npm install --legacy-peer-deps
npm run dev -- --port 5173
```

Access the clinical application in your browser at:
`http://127.0.0.1:5173`

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | `GET` | Service liveness, model checkpoint, and dataset readiness |
| `/api/info` | `GET` | Model specifications, pathology metadata, and thresholds |
| `/api/samples` | `GET` | Curated intraoral sample images for clinical testing |
| `/api/predict` | `POST` | Pathology probabilities for an uploaded image scan |
| `/api/explain` | `POST` | Grad-CAM spatial activation heatmaps and overlay base64 |
| `/api/analyze-full` | `POST` | Combined single-roundtrip inference and Grad-CAM generation |
| `/api/evaluation/models` | `GET` | Available architectures and comparative benchmark specs |
| `/api/evaluation/summary` | `GET` | Overall metrics, metadata, and dataset integrity summary |
| `/api/evaluation/confusion-matrix` | `GET` | 6×6 raw and normalized confusion matrix for selected model |
| `/api/evaluation/class-metrics` | `GET` | Per-class precision, recall, specificity, and F1-scores |
| `/api/evaluation/misclassified` | `GET` | Gallery of misclassified images with probability distributions |
| `/api/evaluation/export/csv` | `GET` | Downloadable CSV evaluation report |
| `/api/evaluation/export/json` | `GET` | Downloadable JSON evaluation report |

---

## Scientific Transparency & Disclaimer

Dentura AI is designed as a clinical decision-support and educational tool. An automated cryptographic dataset integrity audit identified 359 duplicate image pairs across raw train/test partitions in the benchmark dataset, including label discordance between early Gingivitis and Calculus. Prospective clinical deployment requires rigorous deduplication and multi-center validation on calibrated intraoral camera sensors.
