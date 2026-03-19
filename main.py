"""
Panel Pulso Cuanti — API de Inferencia ML
==========================================
FastAPI backend que sirve los 12 modelos entrenados.
Deploy: Railway / Render / cualquier servicio con Python.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import json
import joblib
import numpy as np
from pathlib import Path

# ============================================================
# LOAD MODELS ON STARTUP
# ============================================================

MODEL_DIR = Path(__file__).parent / "models"

models = {}
for f in MODEL_DIR.glob("*_model.joblib"):
    name = f.stem.replace("_model", "")
    models[name] = joblib.load(f)

label_encoders = joblib.load(MODEL_DIR / "label_encoders.joblib")

with open(MODEL_DIR / "feature_config.json") as f:
    feature_config = json.load(f)

with open(MODEL_DIR / "inference_config.json") as f:
    inference_config = json.load(f)

with open(MODEL_DIR / "training_results.json") as f:
    training_results = json.load(f)

FEATURES_A = feature_config["A"]
FEATURES_B = feature_config["B"]

print(f"Loaded {len(models)} models from {MODEL_DIR}")

# ============================================================
# CODEBOOK (for user-friendly labels)
# ============================================================

CODEBOOK = {
    "SEXO": {1: "Femenino", 2: "Masculino"},
    "EDAD_A": {1: "16-29", 2: "30-49", 3: "50-65", 4: "+65"},
    "NED2": {1: "Primario", 2: "Secundario", 3: "Superior"},
    "nse_sim": {1: "ABC1", 2: "C2", 3: "C3", 4: "D1", 5: "D2E"},
    "NSE_AGRUP": {1: "Alta", 2: "Media", 3: "Baja"},
    "REGION": {0: "GBA", 1: "PBA Interior", 2: "CABA", 3: "Cuyo", 4: "NOA", 5: "NEA", 6: "Pampeana", 7: "Patagonia"},
    "AREA": {1: "Interior", 2: "AMBA"},
    "ESTRATO": {0: "GBA", 1: "CABA", 2: "Grandes aglomerados", 3: "Grandes ciudades", 4: "Resto"},
    "c_medica": {1: "Tiene", 2: "No tiene"},
    "estado": {1: "Trabaja", 2: "Desocupado", 3: "Jubilado", 4: "Inactivo"},
    "GRALES23": {1: "Milei (LLA)", 2: "Bullrich (JxC)", 4: "Massa (UxP)", 6: "Schiaretti", 7: "Bregman (FIT)", 99: "NsNc/No votó"},
    "BALLO23": {1: "Milei", 4: "Massa", 99: "NsNc/No votó"},
    "RESP": {1: "Gestión Milei", 2: "Gestión anterior (Massa/Fernández)", 3: "NsNc"},
    "ECONOMIA": {1: "Milei resuelve los problemas", 2: "Sabe pero necesita tiempo", 3: "No sabe resolver", 4: "NsNc"},
}

VARIABLES_CONSULTABLES = {
    "VOTO25": {"desc": "Intención de voto 2025", "familia": "B"},
    "VOTO25_A": {"desc": "Intención de voto (solo demográficas)", "familia": "A"},
    "GESNAC": {"desc": "Evaluación de gestión Milei", "familia": "A"},
    "MILEI2": {"desc": "Imagen de Milei (Pos/Neg)", "familia": "A"},
    "CFK2": {"desc": "Imagen de CFK (Pos/Neg)", "familia": "A"},
    "MM2": {"desc": "Imagen de Macri (Pos/Neg)", "familia": "A"},
    "PBULL2": {"desc": "Imagen de Bullrich (Pos/Neg)", "familia": "A"},
    "KICI2": {"desc": "Imagen de Kicillof (Pos/Neg)", "familia": "A"},
    "ECOHOY": {"desc": "Percepción económica hoy", "familia": "A"},
    "ECOPROSPE": {"desc": "Prospectiva económica", "familia": "A"},
    "RUMBO": {"desc": "Rumbo del gobierno", "familia": "A"},
    "SENTIMIENTO": {"desc": "Sentimiento sobre el futuro", "familia": "B"},
}

# ============================================================
# PYDANTIC MODELS
# ============================================================

class ProfileInput(BaseModel):
    SEXO: Optional[int] = Field(None, description="1=Femenino, 2=Masculino")
    EDAD_A: Optional[int] = Field(None, description="1=16-29, 2=30-49, 3=50-65, 4=+65")
    NED2: Optional[int] = Field(None, description="1=Primario, 2=Secundario, 3=Superior")
    nse_sim: Optional[int] = Field(None, description="1=ABC1, 2=C2, 3=C3, 4=D1, 5=D2E")
    REGION: Optional[int] = Field(None, description="0=GBA, 1=PBA Int, 2=CABA, 3=Cuyo, 4=NOA, 5=NEA, 6=Pampeana, 7=Patagonia")
    AREA: Optional[int] = Field(None, description="1=Interior, 2=AMBA")
    ESTRATO: Optional[int] = Field(None, description="0=GBA, 1=CABA, 2=Gdes aglom, 3=Gdes ciud, 4=Resto")
    c_medica: Optional[int] = Field(None, description="1=Tiene, 2=No tiene")
    estado: Optional[int] = Field(None, description="1=Trabaja, 2=Desocup, 3=Jubilado, 4=Inactivo")
    GRALES23: Optional[int] = Field(None, description="Voto Generales 2023")
    BALLO23: Optional[int] = Field(None, description="Voto Ballotage 2023")
    RESP: Optional[int] = Field(None, description="1=Culpa Milei, 2=Culpa anteriores, 3=NsNc")
    ECONOMIA: Optional[int] = Field(None, description="1=Resuelve, 2=Necesita tiempo, 3=No sabe, 4=NsNc")
    targets: Optional[list[str]] = Field(None, description="Lista de targets a predecir. None = todos")

class ScenarioInput(BaseModel):
    perfil: ProfileInput
    variable: str = Field(..., description="Variable a modificar")
    valor_nuevo: int = Field(..., description="Nuevo valor de la variable")
    target: str = Field(..., description="Target a medir")

# ============================================================
# CORE PREDICTION LOGIC
# ============================================================

def build_feature_vector(profile: dict, features: list[str]) -> np.ndarray:
    """Build numpy feature vector from profile dict."""
    # Compute interactions
    sexo = profile.get("SEXO", 998)
    edad = profile.get("EDAD_A", 998)
    nse = profile.get("nse_sim", 998)
    region = profile.get("REGION", 998)

    profile["SEXO_EDAD"] = sexo * 10 + edad
    profile["REGION_NSE"] = region * 10 + nse
    profile["EDAD_NSE"] = edad * 10 + nse
    profile.setdefault("OLEADA_NUM", 5)

    for feat in features:
        profile.setdefault(feat, 998)

    return np.array([[profile.get(f, 998) for f in features]])


def predict_single(profile: dict, model_name: str) -> dict:
    """Run prediction for one model."""
    if model_name not in models:
        return {"error": f"Modelo '{model_name}' no encontrado"}

    cfg = inference_config[model_name]
    features = FEATURES_B if cfg["feat"] == "B" else FEATURES_A

    X = build_feature_vector(profile.copy(), features)
    model = models[model_name]
    le = label_encoders[model_name]

    proba = model.predict_proba(X)[0]
    classes = le.classes_
    labels = cfg["labels"]

    resultados = []
    for i, cls in enumerate(classes):
        label = labels.get(str(cls), str(cls))
        resultados.append({
            "categoria": label,
            "codigo": int(cls),
            "porcentaje": round(float(proba[i]) * 100, 1),
        })

    resultados.sort(key=lambda x: -x["porcentaje"])

    tr = training_results.get(model_name, {})
    return {
        "modelo": model_name,
        "descripcion": cfg.get("desc", cfg.get("target", "")),
        "resultados": resultados,
        "accuracy_modelo": tr.get("accuracy"),
        "n_entrenamiento": tr.get("n_samples"),
        "familia_features": cfg["feat"],
    }


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Panel Pulso Cuanti API",
    description="API de inferencia ML para opinión pública argentina. 12 modelos, 10,385 casos.",
    version="4.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "servicio": "Panel Pulso Cuanti API",
        "version": "4.0.0",
        "modelos": len(models),
        "endpoints": ["/predict/profile", "/predict/scenario", "/models", "/codebook"],
    }


@app.get("/models")
def list_models():
    """Lista todos los modelos disponibles con sus métricas."""
    result = {}
    for name, cfg in VARIABLES_CONSULTABLES.items():
        tr = training_results.get(name, {})
        result[name] = {
            "descripcion": cfg["desc"],
            "familia": cfg["familia"],
            "accuracy": tr.get("accuracy"),
            "f1_macro": tr.get("f1_macro"),
            "n_samples": tr.get("n_samples"),
            "n_classes": tr.get("n_classes"),
        }
    return result


@app.get("/codebook")
def get_codebook():
    """Devuelve el diccionario de códigos para armar consultas."""
    return CODEBOOK


@app.post("/predict/profile")
def predict_profile(profile: ProfileInput):
    """
    Predecir distribuciones para un perfil dado.
    Enviar solo las variables que se conocen; las demás se tratan como desconocidas.
    """
    profile_dict = {k: v for k, v in profile.model_dump().items() if v is not None and k != "targets"}

    targets = profile.targets or list(VARIABLES_CONSULTABLES.keys())

    predictions = {}
    for target in targets:
        if target not in models:
            continue
        pred = predict_single(profile_dict, target)
        predictions[target] = pred

    # Build human-readable profile description
    desc_parts = []
    for var, val in profile_dict.items():
        if var in CODEBOOK and val in CODEBOOK[var]:
            desc_parts.append(f"{var}={CODEBOOK[var][val]}")
        elif var in CODEBOOK:
            desc_parts.append(f"{var}={val}")
    perfil_desc = ", ".join(desc_parts) if desc_parts else "Nacional (sin filtros)"

    return {
        "perfil": perfil_desc,
        "variables_enviadas": profile_dict,
        "predicciones": predictions,
        "nota": "Los porcentajes reflejan la probabilidad estimada por los modelos ML entrenados con 10,385 casos reales.",
    }


@app.post("/predict/scenario")
def predict_scenario(scenario: ScenarioInput):
    """
    Comparar predicciones entre un perfil base y una versión modificada.
    Devuelve el delta (impacto del cambio).
    """
    if scenario.target not in models:
        raise HTTPException(status_code=400, detail=f"Target '{scenario.target}' no disponible")

    if scenario.variable not in ProfileInput.model_fields:
        raise HTTPException(status_code=400, detail=f"Variable '{scenario.variable}' no válida")

    base_dict = {k: v for k, v in scenario.perfil.model_dump().items() if v is not None and k != "targets"}

    # Baseline prediction
    baseline = predict_single(base_dict.copy(), scenario.target)

    # Modified prediction
    modified_dict = base_dict.copy()
    modified_dict[scenario.variable] = scenario.valor_nuevo
    modified = predict_single(modified_dict, scenario.target)

    # Compute delta
    base_map = {r["categoria"]: r["porcentaje"] for r in baseline["resultados"]}
    mod_map = {r["categoria"]: r["porcentaje"] for r in modified["resultados"]}
    all_cats = list(dict.fromkeys(list(base_map.keys()) + list(mod_map.keys())))

    delta = []
    for cat in all_cats:
        b = base_map.get(cat, 0)
        m = mod_map.get(cat, 0)
        d = round(m - b, 1)
        delta.append({"categoria": cat, "baseline": b, "escenario": m, "delta_pp": d})

    delta.sort(key=lambda x: -abs(x["delta_pp"]))

    # Labels for the change
    var_label = CODEBOOK.get(scenario.variable, {})
    old_val = base_dict.get(scenario.variable)
    old_label = var_label.get(old_val, str(old_val)) if old_val is not None else "no definido"
    new_label = var_label.get(scenario.valor_nuevo, str(scenario.valor_nuevo))

    return {
        "variable_modificada": scenario.variable,
        "cambio": f"{old_label} → {new_label}",
        "target": scenario.target,
        "baseline": baseline,
        "escenario": modified,
        "delta": delta,
        "nota": "Delta en puntos porcentuales. Basado en correlaciones observadas, no causalidad.",
    }


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": len(models)}
