from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
from pydantic import BaseModel
from typing import List, Optional
import os
import pickle
 
app = FastAPI()
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://senia2-0.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
 
# ── Modelo estático ──────────────────────────────────────────────────────────
try:
    modelo_estatico = joblib.load("sign_language_model.pkl")
    print("✅ Modelo estático cargado")
except Exception as e:
    print(f"❌ Error modelo estático: {e}")
    modelo_estatico = None
 
# ── Modelo dinámico ──────────────────────────────────────────────────────────
try:
    import tensorflow as tf
    modelo_dinamico = tf.keras.models.load_model("modelo_lsm.keras")
    with open("label_encoder.pkl", "rb") as f:
        le_dinamico = pickle.load(f)
    print("✅ Modelo dinámico cargado")
except Exception as e:
    print(f"❌ Error modelo dinámico: {e}")
    modelo_dinamico = None
    le_dinamico = None
 
# ── Labels ───────────────────────────────────────────────────────────────────
LETRAS_ESTATICAS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I",
    "L", "M", "N", "O", "P", "R", "S", "T", "U", "V", "W", "Y"
]
 
# ── Modelos de datos ─────────────────────────────────────────────────────────
class DatosMano(BaseModel):
    # Frame actual: mínimo 63 puntos (1 mano) o 135 (2 manos + cara)
    puntos: List[float]
    # Últimos 20 frames para el modelo dinámico (opcional)
    secuencia: Optional[List[List[float]]] = None
 
# ── Normalización para el modelo estático ────────────────────────────────────
def normalizar_puntos_estatico(puntos_lista: list) -> list:
    """
    Normaliza los primeros 63 puntos (mano principal, 21 landmarks × 3).
    Invariante a traslación y escala, igual que durante el entrenamiento.
    """
    pts = np.array(puntos_lista[:63], dtype=np.float32).reshape(21, 3)
    muneca = pts[0]
    pts_centrados = pts - muneca
    distancias = np.linalg.norm(pts_centrados[1:], axis=1)
    tamano = float(np.max(distancias)) if np.max(distancias) > 0 else 1.0
    return (pts_centrados / tamano).flatten().tolist()
 
# ── Endpoint principal ───────────────────────────────────────────────────────
@app.post("/predecir")
async def predecir(entrada: DatosMano):
    res_estatico = None
    res_dinamico = None
 
    # — Modelo estático (corre siempre que haya al menos 63 puntos) —
    if modelo_estatico is not None and len(entrada.puntos) >= 63:
        try:
            puntos_norm = normalizar_puntos_estatico(entrada.puntos)
            datos = np.array(puntos_norm).reshape(1, -1)
            pred = modelo_estatico.predict(datos)
            proba = modelo_estatico.predict_proba(datos)
            res_estatico = {
                "indice": int(pred[0]),
                "confianza": round(float(np.max(proba[0])) * 100, 2),
            }
        except Exception as e:
            print(f"Error estático: {e}")
 
    # — Modelo dinámico (solo cuando el buffer llega a 20 frames) —
    if (
        modelo_dinamico is not None
        and le_dinamico is not None
        and entrada.secuencia is not None
        and len(entrada.secuencia) == 20
    ):
        try:
            seq = np.array(entrada.secuencia, dtype=np.float32)
            seq = np.expand_dims(seq, axis=0)  # shape: (1, 20, 135)
            pred = modelo_dinamico.predict(seq, verbose=0)
            confianza = round(float(np.max(pred[0])) * 100, 2)
            label = str(le_dinamico.inverse_transform([np.argmax(pred[0])])[0])
            res_dinamico = {"label": label, "confianza": confianza}
        except Exception as e:
            print(f"Error dinámico: {e}")
 
    # — Selección automática —
    # El dinámico gana si tiene >70% de confianza Y supera al estático
    usar_dinamico = (
        res_dinamico is not None
        and res_dinamico["confianza"] > 70
        and (
            res_estatico is None
            or res_dinamico["confianza"] >= res_estatico["confianza"]
        )
    )
 
    if usar_dinamico:
        return {
            "indice": -1,                           # el frontend usa "signo" directamente
            "confianza": res_dinamico["confianza"],
            "signo": res_dinamico["label"],
            "modelo": "dinamico",
        }
 
    if res_estatico is not None:
        signo = (
            LETRAS_ESTATICAS[res_estatico["indice"]]
            if res_estatico["indice"] < len(LETRAS_ESTATICAS)
            else "?"
        )
        return {
            "indice": res_estatico["indice"],
            "confianza": res_estatico["confianza"],
            "signo": signo,
            "modelo": "estatico",
        }
 
    return {"indice": -1, "confianza": 0, "signo": "", "modelo": "ninguno"}
 
 
@app.get("/")
async def root():
    return {
        "mensaje": "API SeñIA — Dual Model",
        "modelo_estatico": modelo_estatico is not None,
        "modelo_dinamico": modelo_dinamico is not None,
    }
 
 
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
