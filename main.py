
from fastapi import FastAPI
from models import (
    GeneticoRequest,
    GeneticoResponse,
    ArbolDecisionRequest,
    ArbolDecisionResponse
    GeneticoResponse,
    ClusterRequest,
    ClusterResponse
)
from services import ejecutar_algoritmo_genetico, evaluar_riesgo_cardiaco
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


#Arbol de decision:
@app.post("/arbol_decision", response_model=ArbolDecisionResponse)
def decision_tree_algoritmo(parametro: ArbolDecisionRequest):

    return evaluar_riesgo_cardiaco(
        edad=parametro.edad,
        presion_sistolica=parametro.presion_sistolica,
        colesterol=parametro.colesterol,
        frecuencia_max=parametro.frecuencia_max,
        angina_ejercicio=parametro.angina_ejercicio
    )


# K-MEANS
@app.post("/cluster", response_model=ClusterResponse)
def cluster_algoritmo(parametro: ClusterRequest):
    return ejecutar_kmeans(
        k_clusters=parametro.k_clusters,
        max_iteraciones=parametro.max_iteraciones,
        datos=parametro.datos
    )