
from fastapi import FastAPI
from models import (
    GeneticoRequest,
    GeneticoResponse,
    ClusterRequest,
    ClusterResponse
)
from services import ejecutar_algoritmo_genetico, ejecutar_kmeans

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


# K-MEANS
@app.post("/cluster", response_model=ClusterResponse)
def cluster_algoritmo(parametro: ClusterRequest):
    return ejecutar_kmeans(
        k_clusters=parametro.k_clusters,
        max_iteraciones=parametro.max_iteraciones,
        datos=parametro.datos
    )