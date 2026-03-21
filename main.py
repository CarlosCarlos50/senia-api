from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
from pydantic import BaseModel
from typing import List
import os

app = FastAPI()

# CORS - Permite GitHub Pages (ajusta según tu URL real)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[https://carloscarlos50.github.io/],  # Cambia si usas otro dominio
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar modelo
try:
    modelo = joblib.load("sign_language_model.pkl")
    print("✅ Modelo cargado correctamente")
except Exception as e:
    print(f"❌ Error al cargar el modelo: {e}")
    modelo = None


class DatosMano(BaseModel):
    puntos: List[float]


def normalizar_puntos(puntos_lista):
    """
    Normaliza los 63 puntos (21 landmarks * 3 coordenadas)
    para que sean invariantes a traslación y escala.
    """
    pts = np.array(puntos_lista, dtype=np.float32).reshape(21, 3)
    # Centrar en la muñeca (landmark 0)
    muneca = pts[0]
    pts_centrados = pts - muneca
    # Calcular tamaño de la mano (distancia máxima desde muñeca)
    distancias = np.linalg.norm(pts_centrados[1:], axis=1)
    tamaño = np.max(distancias) if np.max(distancias) > 0 else 1.0
    # Escalar
    pts_normalizados = pts_centrados / tamaño
    # Aplanar de vuelta a 63
    return pts_normalizados.flatten().tolist()


@app.post("/predecir")
async def predecir(entrada: DatosMano):
    if modelo is None:
        raise HTTPException(status_code=500, detail="Modelo no disponible")

    try:
        if len(entrada.puntos) != 63:
            raise HTTPException(
                status_code=400,
                detail=f"Se esperaban 63 puntos, se recibieron {len(entrada.puntos)}",
            )

        # Normalizar los puntos ANTES de pasarlos al modelo
        puntos_normalizados = normalizar_puntos(entrada.puntos)
        datos = np.array(puntos_normalizados).reshape(1, -1)

        prediccion = modelo.predict(datos)
        resultado = int(prediccion[0])

        # Obtener probabilidades
        probabilidades = modelo.predict_proba(datos)
        confianza = float(np.max(probabilidades[0])) * 100  # Convertir a porcentaje

        print(f"Predicción realizada: Índice {resultado}, Confianza: {confianza:.2f}%")
        return {"indice": resultado, "confianza": round(confianza, 2)}

    except Exception as e:
        print(f"Error en proceso: {e}")
        return {"indice": -1, "confianza": 0}


@app.get("/")
async def root():
    return {
        "mensaje": "API de SeñIA con normalización",
        "modelo_cargado": modelo is not None,
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
