
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