import os
import io
import json
import base64
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from PIL import Image

from model_service import (
    service,
    MODEL_BENCHMARKS,
    DISEASE_METADATA,
    CONFIDENCE_THRESHOLD,
    SAMPLES_DIR
)
from evaluation.evaluate_model import evaluator, DATASET_TEST_DIR

app = FastAPI(
    title="Dentura AI — Oral Pathology Diagnostics API",
    description="Clinical Oral Pathology Classification & Comprehensive Model Evaluation Service",
    version="2.1.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Dentura AI",
        "modelLoaded": service.model is not None,
        "gradcamReady": service.grad_model is not None,
        "evaluationReady": os.path.exists(DATASET_TEST_DIR)
    }

@app.get("/api/info")
def get_model_info():
    return {
        "projectName": "Dentura AI",
        "architecture": "EfficientNetB0 Deep Transfer Classifier with Grad-CAM",
        "benchmarks": MODEL_BENCHMARKS,
        "classes": service.class_names,
        "diseaseMetadata": DISEASE_METADATA,
        "confidenceThreshold": CONFIDENCE_THRESHOLD
    }

@app.get("/api/samples")
def get_sample_images():
    return service.get_sample_images()

@app.get("/api/samples/{sample_id}/raw")
def get_sample_raw(sample_id: str):
    fpath = os.path.join(SAMPLES_DIR, sample_id)
    if not os.path.exists(fpath):
        raise HTTPException(status_code=404, detail="Sample image not found")
    with open(fpath, "rb") as f:
        data = f.read()
    return Response(content=data, media_type="image/jpeg")

def _load_image_from_bytes(file_bytes: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(file_bytes))
        img.verify()
        return Image.open(io.BytesIO(file_bytes))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")

@app.post("/api/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    content = await file.read()
    image = _load_image_from_bytes(content)

    result = service.predict(image)
    return JSONResponse(content=result)

@app.post("/api/explain")
async def explain_endpoint(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    content = await file.read()
    image = _load_image_from_bytes(content)

    gradcam_res = service.generate_gradcam(image)
    return JSONResponse(content=gradcam_res)

@app.post("/api/analyze-full")
async def analyze_full_endpoint(file: UploadFile = File(...)):
    """Returns inference predictions and Grad-CAM in one request."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    content = await file.read()
    image = _load_image_from_bytes(content)

    pred_res = service.predict(image)
    gradcam_res = service.generate_gradcam(image)

    return JSONResponse(content={
        "prediction": pred_res,
        "gradcam": gradcam_res
    })


# Evaluation endpoints
@app.get("/api/evaluation/models")
def get_available_models():
    """Returns list of all available classification models."""
    return evaluator.get_models_list()

@app.get("/api/evaluation/summary")
def get_evaluation_summary(model: str = Query("efficientnet_b0", description="Model identifier")):
    eval_data = evaluator.run_evaluation(model_id=model)
    if not eval_data.get("available"):
        raise HTTPException(status_code=503, detail=eval_data.get("error", "Evaluation unavailable"))

    return {
        "available": True,
        "modelId": eval_data.get("modelId", model),
        "metadata": eval_data["metadata"],
        "overallMetrics": eval_data["overallMetrics"],
        "integrityAudit": eval_data["integrityAudit"],
        "scientificDisclaimer": eval_data["scientificDisclaimer"]
    }

@app.get("/api/evaluation/confusion-matrix")
def get_confusion_matrix(model: str = Query("efficientnet_b0", description="Model identifier")):
    eval_data = evaluator.run_evaluation(model_id=model)
    if not eval_data.get("available"):
        raise HTTPException(status_code=503, detail=eval_data.get("error", "Evaluation unavailable"))

    return {
        "available": True,
        "modelId": eval_data.get("modelId", model),
        "confusionMatrix": eval_data["confusionMatrix"]
    }

@app.get("/api/evaluation/class-metrics")
def get_class_metrics(model: str = Query("efficientnet_b0", description="Model identifier")):
    eval_data = evaluator.run_evaluation(model_id=model)
    if not eval_data.get("available"):
        raise HTTPException(status_code=503, detail=eval_data.get("error", "Evaluation unavailable"))

    return {
        "available": True,
        "modelId": eval_data.get("modelId", model),
        "classMetrics": eval_data["perClassMetrics"],
        "classOrder": eval_data["metadata"]["classOrder"]
    }

@app.get("/api/evaluation/misclassified")
def get_misclassified_samples(
    model: str = Query("efficientnet_b0", description="Model identifier"),
    true_class: Optional[str] = Query(None, description="Filter by true label"),
    predicted_class: Optional[str] = Query(None, description="Filter by predicted label"),
    limit: int = Query(24, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    eval_data = evaluator.run_evaluation(model_id=model)
    if not eval_data.get("available"):
        raise HTTPException(status_code=503, detail=eval_data.get("error", "Evaluation unavailable"))

    samples = eval_data.get("misclassifiedSamples", [])

    # Filter by true class
    if true_class and true_class != "all":
        samples = [s for s in samples if s["trueClass"].lower() == true_class.lower()]

    # Filter by predicted class
    if predicted_class and predicted_class != "all":
        samples = [s for s in samples if s["predictedClass"].lower() == predicted_class.lower()]

    total_filtered = len(samples)
    paginated = samples[offset : offset + limit]

    return {
        "available": True,
        "modelId": eval_data.get("modelId", model),
        "total": total_filtered,
        "offset": offset,
        "limit": limit,
        "samples": paginated
    }

@app.get("/api/evaluation/image/{class_name}/{filename}")
def get_test_image(class_name: str, filename: str):
    """
    Securely serves a test set image without exposing absolute filesystem paths.
    Validates class name and prevents path traversal.
    """
    if class_name not in evaluator.class_names:
        raise HTTPException(status_code=400, detail="Invalid class name")

    # Sanitize filename
    safe_filename = os.path.basename(filename)
    fpath = os.path.join(DATASET_TEST_DIR, class_name, safe_filename)

    if not os.path.exists(fpath):
        raise HTTPException(status_code=404, detail="Test image not found")

    with open(fpath, "rb") as f:
        data = f.read()

    media_type = "image/png" if safe_filename.lower().endswith(".png") else "image/jpeg"
    return Response(content=data, media_type=media_type)

@app.get("/api/evaluation/export/csv")
def export_csv(model: str = Query("efficientnet_b0", description="Model identifier")):
    csv_content = evaluator.export_metrics_csv(model_id=model)
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=dentura_ai_{model}_evaluation.csv"}
    )

@app.get("/api/evaluation/export/json")
def export_json(model: str = Query("efficientnet_b0", description="Model identifier")):
    eval_data = evaluator.run_evaluation(model_id=model)
    json_bytes = json.dumps(eval_data, indent=2).encode("utf-8")
    return Response(
        content=json_bytes,
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename=dentura_ai_{model}_evaluation_report.json"}
    )

@app.post("/api/evaluation/run")
def trigger_evaluation_run():
    """Forces recalculation of the model evaluation and updates cache."""
    try:
        updated_results = evaluator.run_evaluation(force_refresh=True)
        return {
            "status": "success",
            "message": "Model evaluation refreshed successfully.",
            "overallMetrics": updated_results.get("overallMetrics")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8008, reload=False)
