import os
from typing import Dict, Tuple

import cv2
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

try:
    import joblib  # type: ignore
except Exception:  # pragma: no cover
    joblib = None  # type: ignore

from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern


CLASSES = ["Normal", "Pneumonia", "COVID"]
MODEL_KEYS = ["svm", "rf", "knn"]


def _load_pickle(path: str):
    if joblib is not None:
        return joblib.load(path)
    import pickle

    with open(path, "rb") as f:
        return pickle.load(f)


def preprocess_image_bgr_to_gray_norm(img_bgr: np.ndarray) -> np.ndarray:
    if img_bgr is None or img_bgr.size == 0:
        raise ValueError("Invalid image.")

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (224, 224), interpolation=cv2.INTER_AREA)
    gray = gray.astype(np.float32) / 255.0
    return gray


def extract_glcm_features(gray_norm: np.ndarray, levels: int = 16) -> np.ndarray:
    img_uint8 = np.clip(gray_norm * 255.0, 0, 255).astype(np.uint8)
    step = max(1, 256 // levels)
    img_q = (img_uint8 // step).astype(np.uint8)
    img_q = np.clip(img_q, 0, levels - 1)

    distances = [1, 2, 3]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    glcm = graycomatrix(
        img_q,
        distances=distances,
        angles=angles,
        levels=levels,
        symmetric=True,
        normed=True,
    )

    props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
    feats = []
    for p in props:
        v = graycoprops(glcm, p)  # shape: (len(distances), len(angles))
        feats.append(v.mean())
        feats.append(v.std())
    return np.array(feats, dtype=np.float32)


def extract_lbp_features(gray_norm: np.ndarray, radius: int = 2, n_points: int = 16) -> np.ndarray:
    """
    LBP configuration aligned to training: uniform LBP with P=16 -> (P + 2) = 18 bins.
    This matches the saved scaler/models feature size (26274).
    """
    img_uint8 = np.clip(gray_norm * 255.0, 0, 255).astype(np.uint8)
    lbp = local_binary_pattern(img_uint8, n_points, radius, method="uniform")
    n_bins = n_points + 2
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_bins + 1), range=(0, n_bins))
    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-8)
    return hist


def extract_hog_features(gray_norm: np.ndarray) -> np.ndarray:
    feats = hog(
        gray_norm,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=False,
        feature_vector=True,
    )
    return feats.astype(np.float32)


def extract_features(gray_norm: np.ndarray) -> np.ndarray:
    glcm_f = extract_glcm_features(gray_norm)
    lbp_f = extract_lbp_features(gray_norm)
    hog_f = extract_hog_features(gray_norm)
    return np.concatenate([glcm_f, lbp_f, hog_f], axis=0)


def _safe_proba_dict(proba_vec: np.ndarray) -> Dict[str, float]:
    proba_vec = np.asarray(proba_vec).reshape(-1)
    out: Dict[str, float] = {}
    for i, name in enumerate(CLASSES):
        out[name] = float(proba_vec[i]) if i < proba_vec.shape[0] else 0.0
    return out


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    x = x - float(np.max(x))
    e = np.exp(x)
    return e / (float(np.sum(e)) + 1e-8)


def predict_with_model(model, x_scaled: np.ndarray) -> Tuple[int, str, Dict[str, float] | None]:
    pred = model.predict(x_scaled)
    idx = int(np.asarray(pred).reshape(-1)[0])
    label = CLASSES[idx] if 0 <= idx < len(CLASSES) else str(idx)

    proba = None
    if hasattr(model, "predict_proba"):
        try:
            p = model.predict_proba(x_scaled)[0]
            proba = _safe_proba_dict(p)
        except Exception:
            proba = None
    elif hasattr(model, "decision_function"):
        try:
            df = model.decision_function(x_scaled)
            df = np.asarray(df)
            if df.ndim == 1 and df.shape[0] == len(CLASSES):
                p = _softmax(df)
                proba = _safe_proba_dict(p)
            elif df.ndim == 2 and df.shape[1] == len(CLASSES):
                p = _softmax(df[0])
                proba = _safe_proba_dict(p)
        except Exception:
            proba = None
    return idx, label, proba


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)

    here = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.abspath(os.path.join(here, "..", "model"))
    scaler_candidates = [
        os.path.join(model_dir, "scaler.pkl"),
        os.path.join(model_dir, "scaler (1).pkl"),
    ]
    scaler_path = next((p for p in scaler_candidates if os.path.exists(p)), scaler_candidates[0])
    scaler = _load_pickle(os.path.abspath(scaler_path))

    def load_models() -> Dict[str, object]:
        models: Dict[str, object] = {}
        # User-selectable models (as saved from Colab).
        paths = {
            "svm": os.path.join(model_dir, "svm_model.pkl"),
            "rf": os.path.join(model_dir, "rf_model.pkl"),
            "knn": os.path.join(model_dir, "knn_model.pkl"),
        }
        for k, p in paths.items():
            if os.path.exists(p):
                try:
                    models[k] = _load_pickle(p)
                except Exception:
                    pass
        return models

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"}), 200

    @app.post("/predict")
    def predict():
        models = load_models()
        if not models:
            return (
                jsonify(
                    {
                        "error": "No models found in model/.",
                        "expected_files": ["svm_model.pkl", "rf_model.pkl", "knn_model.pkl"],
                    }
                ),
                500,
            )

        file = request.files.get("file") or request.files.get("image")
        if file is None:
            return jsonify({"error": "No image uploaded. Use form-data key 'file'."}), 400

        data = file.read()
        if not data:
            return jsonify({"error": "Empty file."}), 400

        img_arr = np.frombuffer(data, dtype=np.uint8)
        img_bgr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return jsonify({"error": "Could not decode image. Upload a valid PNG/JPG."}), 400

        requested = (
            (request.form.get("model") or request.args.get("model") or "all")
            .strip()
            .lower()
        )
        if requested == "all":
            selected_keys = [k for k in MODEL_KEYS if k in models]
        else:
            selected_keys = [requested]
        selected_keys = [k for k in selected_keys if k in models]
        if not selected_keys:
            return (
                jsonify(
                    {
                        "error": "Requested model not available.",
                        "available_models": sorted(list(models.keys())),
                    }
                ),
                400,
            )

        try:
            gray_norm = preprocess_image_bgr_to_gray_norm(img_bgr)
            feats = extract_features(gray_norm)
            x = feats.reshape(1, -1)
            x_scaled = scaler.transform(x)
        except Exception as e:
            return jsonify({"error": "Prediction failed.", "details": str(e)}), 500

        results: Dict[str, Dict[str, object]] = {}
        for k in selected_keys:
            try:
                idx, label, proba = predict_with_model(models[k], x_scaled)
                r: Dict[str, object] = {"class_index": idx, "class_name": label}
                if proba is not None:
                    r["probabilities"] = proba
                results[k] = r
            except Exception as e:
                results[k] = {"error": str(e)}

        # Simple "insights": average probabilities across models (when available)
        probs_list = [r.get("probabilities") for r in results.values() if isinstance(r, dict)]
        probs_list = [p for p in probs_list if isinstance(p, dict)]
        insights = None
        if probs_list:
            avg: Dict[str, float] = {c: 0.0 for c in CLASSES}
            for p in probs_list:
                for c in CLASSES:
                    avg[c] += float(p.get(c, 0.0))
            for c in CLASSES:
                avg[c] /= max(1, len(probs_list))
            insights = {"average_probabilities": avg}

        resp = {
            "classes": CLASSES,
            "available_models": sorted(list(models.keys())),
            "results": results,
        }
        if insights is not None:
            resp["feature_insights"] = insights
        return jsonify(resp), 200

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

