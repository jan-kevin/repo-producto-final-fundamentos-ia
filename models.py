
# Dependencias:
from typing import List, Literal
from pydantic import BaseModel, Field


# ==========================================
# ALGORITMO GENETICO
# ==========================================

class GeneticoRequest(BaseModel):
    mensaje_oculto: str
    tam_poblacion: int = Field(gt=10, le=500)
    max_generaciones: int = Field(gt=1, le=1000)
    prob_cruce: float = Field(ge=0.0, le=1.0)
    prob_mutacion: float = Field(ge=0.0, le=0.5)


class GeneticoResponse(BaseModel):
    estado: Literal["exitoso"]
    mensaje_oculto_objetivo: str
    solucion_obtenida: str
    aptitud_alcanzada: int
    generaciones_ejecutadas: int
    convergencia: bool
    historial_fitness_max: List[int]

# ==========================================
# ARBOL DE DECISIO
# ==========================================

class ArbolDecisionRequest(BaseModel):
    edad: int = Field(ge=18, le=100)
    presion_sistolica: int = Field(ge=80, le=220)
    colesterol: int = Field(ge=100, le=500)
    frecuencia_max: int = Field(ge=60, le=220)
    angina_ejercicio: int = Field(ge=0, le=1)


class PacienteEvaluado(BaseModel):
    edad: int
    presion_sistolica: int
    colesterol: int
    frecuencia_max: int
    angina_ejercicio: int


class ArbolDecisionResponse(BaseModel):
    estado: Literal["exitoso"]
    paciente_evaluado: PacienteEvaluado
    clase_predicha: int
    diagnostico_texto: str
    probabilidad_riesgo: float
    nivel_riesgo: Literal["Bajo", "Medio", "Alto"]