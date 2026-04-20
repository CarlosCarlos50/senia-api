from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
from pydantic import BaseModel
from typing import List, Optional
import os
import pickle
import tensorflow as tf

app = FastAPI()

# ========== CORS ==========
# Permite los orígenes necesarios (agrega el dominio de Vercel)
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

# ... el resto de tu código (carga de modelos, endpoints, etc.) no cambia

# ========== MODELO ESTÁTICO ==========
try:
    modelo_estatico = joblib.load("sign_language_model.pkl")
    print("✅ Modelo estático cargado")
except Exception as e:
    print(f"❌ Error modelo estático: {e}")
    modelo_estatico = None

# ========== MODELO DINÁMICO (SavedModel) ==========
modelo_dinamico = None
le_dinamico = None
try:
    # Forzar CPU (evita warnings de GPU)
    tf.config.set_visible_devices([], 'GPU')
    # Cargar el SavedModel (cambia "modelo_lsm" por el nombre de tu carpeta)
    modelo_dinamico = tf.saved_model.load("modelo_lsm_savedmodel")
    # Obtener la firma por defecto
    infer = modelo_dinamico.signatures['serving_default']
    print("✅ Modelo dinámico (SavedModel) cargado")
    print("Firma de entrada:", infer.structured_input_signature)
    print("Firma de salida:", infer.structured_outputs)
    
    with open("label_encoder.pkl", "rb") as f:
        le_dinamico = pickle.load(f)
except Exception as e:
    print(f"❌ Error cargando modelo dinámico: {e}")
    modelo_dinamico = None
    le_dinamico = None

# ========== CONSTANTES ==========
LETRAS_ESTATICAS = ["A","B","C","D","E","F","G","H","I","L","M","N","O","P","R","S","T","U","V","W","Y"]

# ========== NORMALIZACIÓN ==========
def normalizar_puntos_estatico(puntos_lista: list) -> list:
    pts = np.array(puntos_lista[:63], dtype=np.float32).reshape(21, 3)
    muneca = pts[0]
    pts_centrados = pts - muneca
    distancias = np.linalg.norm(pts_centrados[1:], axis=1)
    tamano = float(np.max(distancias)) if np.max(distancias) > 0 else 1.0
    return (pts_centrados / tamano).flatten().tolist()

# ========== MODELOS DE DATOS ==========
class DatosMano(BaseModel):
    puntos: List[float]               # 63 o 136 puntos
    secuencia: Optional[List[List[float]]] = None   # 20 frames de 136 puntos

# ========== ENDPOINT PRINCIPAL ==========
@app.post("/predecir")
async def predecir(entrada: DatosMano):
    res_estatico = None
    res_dinamico = None

    # --- Estático ---
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

    # --- Dinámico (SavedModel) ---
    if (modelo_dinamico is not None and le_dinamico is not None and
        entrada.secuencia is not None and len(entrada.secuencia) == 20):
        try:
            seq = np.array(entrada.secuencia, dtype=np.float32)  # (20, 136)
            seq = np.expand_dims(seq, axis=0)  # (1, 20, 136)
            input_tensor = tf.convert_to_tensor(seq)
            
            # Llamar al SavedModel. Algunos modelos esperan un diccionario con el nombre de la entrada.
            # Obtenemos el nombre de la primera entrada de la firma
            input_name = list(infer.structured_input_signature[1].keys())[0]
            predictions = infer(**{input_name: input_tensor})
            
            output = next(iter(predictions.values()))
            probs = output.numpy()[0]
            idx = np.argmax(probs)
            confianza = round(float(probs[idx]) * 100, 2)
            label = str(le_dinamico.inverse_transform([idx])[0])
            res_dinamico = {"label": label, "confianza": confianza}
        except Exception as e:
            print(f"Error en dinámico: {e}")

    # --- Lógica de selección ---
    usar_dinamico = (res_dinamico is not None and
                     res_dinamico["confianza"] > 70 and
                     (res_estatico is None or res_dinamico["confianza"] >= res_estatico["confianza"]))

    if usar_dinamico:
        return {
            "indice": -1,
            "confianza": res_dinamico["confianza"],
            "signo": res_dinamico["label"],
            "modelo": "dinamico",
        }

    if res_estatico is not None:
        signo = LETRAS_ESTATICAS[res_estatico["indice"]] if res_estatico["indice"] < len(LETRAS_ESTATICAS) else "?"
        return {
            "indice": res_estatico["indice"],
            "confianza": res_estatico["confianza"],
            "signo": signo,
            "modelo": "estatico",
        }

    return {"indice": -1, "confianza": 0, "signo": "", "modelo": "ninguno"}

# ========== ENDPOINT DE ESTADO ==========
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
