"""
Panel Pulso Cuanti v4.1 — API de Inferencia ML
VOTO25 es predictor, no target.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import json
import joblib
import numpy as np
from pathlib import Path

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
print(f"Loaded {len(models)} models")

CODEBOOK = {
    "SEXO": {1: "Femenino", 2: "Masculino"},
    "EDAD_A": {1: "16-29", 2: "30-49", 3: "50-65", 4: "+65"},
    "NED2": {1: "Primario", 2: "Secundario", 3: "Superior"},
    "nse_sim": {1: "ABC1", 2: "C2", 3: "C3", 4: "D1", 5: "D2E"},
    "REGION": {0: "GBA", 1: "PBA Interior", 2: "CABA", 3: "Cuyo", 4: "NOA", 5: "NEA", 6: "Pampeana", 7: "Patagonia"},
    "AREA": {1: "Interior", 2: "AMBA"},
    "ESTRATO": {0: "GBA", 1: "CABA", 2: "Gdes aglomerados", 3: "Gdes ciudades", 4: "Resto"},
    "c_medica": {1: "Tiene", 2: "No tiene"},
    "estado": {1: "Trabaja", 2: "Desocupado", 3: "Jubilado", 4: "Inactivo"},
    "GRALES23": {1: "Milei (LLA)", 2: "Bullrich (JxC)", 4: "Massa (UxP)", 7: "Bregman (FIT)", 99: "NsNc/No votó"},
    "BALLO23": {1: "Milei", 4: "Massa", 99: "NsNc/No votó"},
    "VOTO25": {1: "LLA", 2: "Fuerza Patria", 3: "Prov. Unidas", 4: "FIT", 5: "Otro", 99: "NsNc/No votó"},
    "RESP": {1: "Gestión Milei", 2: "Gestión anterior", 3: "NsNc"},
    "ECONOMIA": {1: "Resuelve los problemas", 2: "Necesita tiempo", 3: "No sabe resolver", 4: "NsNc"},
}

VARIABLES_INFO = {}
for name, cfg in inference_config.items():
    tr = training_results.get(name, {})
    VARIABLES_INFO[name] = {
        "desc": cfg["desc"], "feat": cfg["feat"],
        "accuracy": tr.get("accuracy"), "f1_macro": tr.get("f1_macro"), "n_samples": tr.get("n_samples"),
    }

class ProfileInput(BaseModel):
    SEXO: Optional[int] = None
    EDAD_A: Optional[int] = None
    NED2: Optional[int] = None
    nse_sim: Optional[int] = None
    REGION: Optional[int] = None
    AREA: Optional[int] = None
    ESTRATO: Optional[int] = None
    c_medica: Optional[int] = None
    estado: Optional[int] = None
    GRALES23: Optional[int] = None
    BALLO23: Optional[int] = None
    VOTO25: Optional[int] = None
    RESP: Optional[int] = None
    ECONOMIA: Optional[int] = None
    targets: Optional[list[str]] = None

class ScenarioInput(BaseModel):
    perfil: ProfileInput
    variable: str
    valor_nuevo: int
    target: str

def build_vector(profile: dict, features: list[str]) -> np.ndarray:
    p = profile.copy()
    p["SEXO_EDAD"] = p.get("SEXO", 998) * 10 + p.get("EDAD_A", 998)
    p["REGION_NSE"] = p.get("REGION", 998) * 10 + p.get("nse_sim", 998)
    p["EDAD_NSE"] = p.get("EDAD_A", 998) * 10 + p.get("nse_sim", 998)
    p["VOTO_CRUCE"] = p.get("BALLO23", 998) * 100 + p.get("VOTO25", 998)
    p.setdefault("OLEADA_NUM", 5)
    for f in features:
        p.setdefault(f, 998)
    return np.array([[p.get(f, 998) for f in features]])

def predict_one(profile: dict, model_name: str) -> dict:
    if model_name not in models:
        return {"error": f"Modelo '{model_name}' no encontrado"}
    cfg = inference_config[model_name]
    X = build_vector(profile.copy(), FEATURES_B if cfg["feat"] == "B" else FEATURES_A)
    le = label_encoders[model_name]
    proba = models[model_name].predict_proba(X)[0]
    resultados = [{"categoria": cfg["labels"].get(str(c), str(c)), "codigo": int(c), "porcentaje": round(float(proba[i]) * 100, 1)} for i, c in enumerate(le.classes_)]
    resultados.sort(key=lambda x: -x["porcentaje"])
    tr = training_results.get(model_name, {})
    return {"modelo": model_name, "descripcion": cfg.get("desc", ""), "resultados": resultados, "accuracy_modelo": tr.get("accuracy"), "n_entrenamiento": tr.get("n_samples"), "familia_features": cfg["feat"]}

app = FastAPI(title="Panel Pulso Cuanti API", version="4.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def root():
    return {"servicio": "Panel Pulso Cuanti API", "version": "4.1.0", "modelos": len(models), "cambio_clave": "VOTO25 es predictor, no target"}

@app.get("/models")
def list_models():
    return VARIABLES_INFO

@app.get("/codebook")
def get_codebook():
    return CODEBOOK

@app.post("/predict/profile")
def predict_profile(profile: ProfileInput):
    profile_dict = {k: v for k, v in profile.model_dump().items() if v is not None and k != "targets"}
    targets = profile.targets or [k for k in inference_config if not k.endswith("_A")]
    predictions = {}
    for t in targets:
        if t in models:
            predictions[t] = predict_one(profile_dict, t)
    desc_parts = [f"{CODEBOOK[v][val]}" for v, val in profile_dict.items() if v in CODEBOOK and val in CODEBOOK[v]]
    return {"perfil": ", ".join(desc_parts) if desc_parts else "Sin filtros", "variables_enviadas": profile_dict, "predicciones": predictions}

@app.post("/predict/scenario")
def predict_scenario(scenario: ScenarioInput):
    if scenario.target not in models:
        raise HTTPException(400, f"Target '{scenario.target}' no disponible")
    base_dict = {k: v for k, v in scenario.perfil.model_dump().items() if v is not None and k != "targets"}
    baseline = predict_one(base_dict.copy(), scenario.target)
    mod_dict = base_dict.copy()
    mod_dict[scenario.variable] = scenario.valor_nuevo
    modified = predict_one(mod_dict, scenario.target)
    bm = {r["categoria"]: r["porcentaje"] for r in baseline["resultados"]}
    mm = {r["categoria"]: r["porcentaje"] for r in modified["resultados"]}
    delta = [{"categoria": c, "baseline": bm.get(c, 0), "escenario": mm.get(c, 0), "delta_pp": round(mm.get(c, 0) - bm.get(c, 0), 1)} for c in dict.fromkeys(list(bm) + list(mm))]
    delta.sort(key=lambda x: -abs(x["delta_pp"]))
    vl = CODEBOOK.get(scenario.variable, {})
    old_v = base_dict.get(scenario.variable)
    return {"variable_modificada": scenario.variable, "cambio": f"{vl.get(old_v, str(old_v))} → {vl.get(scenario.valor_nuevo, str(scenario.valor_nuevo))}", "target": scenario.target, "baseline": baseline, "escenario": modified, "delta": delta}

@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": len(models)}
