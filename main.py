from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import tensorflow as tf
import pickle
from pydantic import BaseModel
from typing import List, Optional
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://senia2-0-uqbm2qwun-carloscarlos50s-projects.vercel.app",
        "https://senia2-0.vercel.app",          # tu dominio principal de Vercel
        # "http://localhost:3000",               # para pruebas locales
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========== MODELO ESTÁTICO ==========
modelo_estatico = None
try:
    modelo_estatico = joblib.load("sign_language_model.pkl")
    print("✅ Modelo estático cargado")
except Exception as e:
    print(f"❌ Error cargando modelo estático: {e}")

# ========== MODELO DINÁMICO (SavedModel) ==========
modelo_dinamico = None
le_dinamico = None
try:
    # Cargar el SavedModel desde la carpeta 'modelo_lsm_savedmodel'
    modelo_dinamico = tf.saved_model.load("modelo_lsm_savedmodel")
    # Obtener la firma por defecto
    infer = modelo_dinamico.signatures['serving_default']
    print("✅ Modelo dinámico (SavedModel) cargado")
    print("Firma de entrada:", infer.structured_input_signature)
    print("Firma de salida:", infer.structured_outputs)

    # Cargar el label encoder
    with open("label_encoder.pkl", "rb") as f:
        le_dinamico = pickle.load(f)
except Exception as e:
    print(f"❌ Error cargando modelo dinámico: {e}")

# ========== CONSTANTES ==========
LETRAS_ESTATICAS = ["A","B","C","D","E","F","G","H","I","L","M","N","O","P","R","S","T","U","V","W","Y"]

# ========== NORMALIZACIÓN (estático) ==========
def normalizar_puntos_estatico(puntos_lista: list) -> list:
    pts = np.array(puntos_lista[:63], dtype=np.float32).reshape(21, 3)
    muneca = pts[0]
    pts_centrados = pts - muneca
    distancias = np.linalg.norm(pts_centrados[1:], axis=1)
    tamano = float(np.max(distancias)) if np.max(distancias) > 0 else 1.0
    return (pts_centrados / tamano).flatten().tolist()

# ========== MODELOS DE DATOS ==========
class DatosMano(BaseModel):
    puntos: List[float]
    secuencia: Optional[List[List[float]]] = None   # 20 frames de 135 puntos

# ========== ENDPOINT PRINCIPAL ==========
@app.post("/predecir")
async def predecir(entrada: DatosMano):
    res_estatico = None
    res_dinamico = None

    # --- Modelo estático ---
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
            print(f"Error en estático: {e}")

    # --- Modelo dinámico (SavedModel) ---
    if (modelo_dinamico is not None and le_dinamico is not None and
        entrada.secuencia is not None and len(entrada.secuencia) == 20):
        try:
            seq = np.array(entrada.secuencia, dtype=np.float32)  # (20, 135)
            seq = np.expand_dims(seq, axis=0)  # (1, 20, 135)
            input_tensor = tf.convert_to_tensor(seq)
            # El nombre del tensor de entrada suele ser 'input_1' (ajusta según logs)
            predictions = infer(input_1=input_tensor)
            output = predictions['output_0'].numpy()[0]
            idx = np.argmax(output)
            confianza = round(float(output[idx]) * 100, 2)
            signo = le_dinamico.inverse_transform([idx])[0]
            res_dinamico = {"signo": signo, "confianza": confianza}
        except Exception as e:
            print(f"Error en dinámico: {e}")

    # --- Selección automática (prioriza dinámico con alta confianza) ---
    if res_dinamico and res_dinamico["confianza"] > 70:
        return {
            "indice": -1,
            "confianza": res_dinamico["confianza"],
            "signo": res_dinamico["signo"],
            "modelo": "dinamico"
        }
    elif res_estatico:
        signo = LETRAS_ESTATICAS[res_estatico["indice"]] if res_estatico["indice"] < len(LETRAS_ESTATICAS) else "?"
        return {
            "indice": res_estatico["indice"],
            "confianza": res_estatico["confianza"],
            "signo": signo,
            "modelo": "estatico"
        }
    return {"indice": -1, "confianza": 0, "signo": "", "modelo": "ninguno"}

@app.get("/")
async def root():
    return {
        "mensaje": "API SeñIA Dual",
        "estatico": modelo_estatico is not None,
        "dinamico": modelo_dinamico is not None
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
