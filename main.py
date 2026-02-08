from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
from pydantic import BaseModel
from typing import List
import os

app = FastAPI()

# CORS - Permitir todo temporalmente, luego lo ajustas
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar modelo
try:
    modelo = joblib.load('sign_language_model.pkl')
    print("✅ Modelo cargado correctamente")
except Exception as e:
    print(f"❌ Error al cargar el modelo: {e}")
    modelo = None

class DatosMano(BaseModel):
    puntos: List[float]

@app.post("/predecir")
async def predecir(entrada: DatosMano):
    if modelo is None:
        raise HTTPException(status_code=500, detail="Modelo no disponible")
    
    try:
        # Validar que vengan 63 puntos
        if len(entrada.puntos) != 63:
            raise HTTPException(
                status_code=400, 
                detail=f"Se esperaban 63 puntos, se recibieron {len(entrada.puntos)}"
            )
        
        datos = np.array(entrada.puntos).reshape(1, -1)
        prediccion = modelo.predict(datos)
        resultado = int(prediccion[0])
        
        print(f"Predicción realizada: Índice {resultado}")
        return {"indice": resultado}
        
    except Exception as e:
        print(f"Error en proceso: {e}")
        return {"indice": -1}

@app.get("/")
async def root():
    return {"mensaje": "API de SeñIA funcionando", "modelo_cargado": modelo is not None}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)