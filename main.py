from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
from pydantic import BaseModel
from typing import List
import os

app = FastAPI()

# CORS - permite tu dominio de Vercel (ajusta si es diferente)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://senia2-0.vercel.app", "https://*.vercel.app", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar modelo
try:
    modelo = joblib.load('sign_language_model.pkl')
    print("✅ Modelo estático cargado correctamente")
except Exception as e:
    print(f"❌ Error al cargar el modelo: {e}")
    modelo = None

class DatosMano(BaseModel):
    puntos: List[float]

def normalizar_puntos(puntos_lista):
    pts = np.array(puntos_lista, dtype=np.float32).reshape(21, 3)
    muñeca = pts[0]
    pts_centrados = pts - muñeca
    distancias = np.linalg.norm(pts_centrados[1:], axis=1)
    tamaño = np.max(distancias) if np.max(distancias) > 0 else 1.0
    pts_normalizados = pts_centrados / tamaño
    return pts_normalizados.flatten().tolist()

@app.post("/predecir")
async def predecir(entrada: DatosMano):
    if modelo is None:
        raise HTTPException(status_code=500, detail="Modelo no disponible")
    try:
        if len(entrada.puntos) != 63:
            raise HTTPException(status_code=400, detail=f"Se esperaban 63 puntos, se recibieron {len(entrada.puntos)}")
        puntos_norm = normalizar_puntos(entrada.puntos)
        datos = np.array(puntos_norm).reshape(1, -1)
        prediccion = modelo.predict(datos)
        resultado = int(prediccion[0])
        proba = modelo.predict_proba(datos)
        confianza = round(float(np.max(proba[0])) * 100, 2)
        return {"indice": resultado, "confianza": confianza}
    except Exception as e:
        print(f"Error en proceso: {e}")
        return {"indice": -1, "confianza": 0}

@app.get("/")
async def root():
    return {"mensaje": "API de SeñIA estática", "modelo_cargado": modelo is not None}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
