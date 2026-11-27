
from fastapi import FastAPI
from models import (
    GeneticoRequest,
    GeneticoResponse
)
from services import ejecutar_algoritmo_genetico

app = FastAPI(
    title="API RESTful de Fundamentos de Inteligencia Artificial",
    description="API para ejecutar algoritmos genéticos, árboles de decisión y K-Means. Producto FInal de la materia Fundamentos de Inteligencia Artificial.",
    version="1.0.0"
)


@app.get("/")
def read_root():
    return {
        "mensaje": "Bienvenido a la API RESTful de IA",
        "estado": "activo"
    }


# Genetico
@app.post("/genetico", response_model=GeneticoResponse)
def genetic_algorithm(parametro: GeneticoRequest):
    return ejecutar_algoritmo_genetico(
        mensaje_oculto=parametro.mensaje_oculto,
        tam_poblacion=parametro.tam_poblacion,
        max_generaciones=parametro.max_generaciones,
        prob_cruce=parametro.prob_cruce,
        prob_mutacion=parametro.prob_mutacion
    )