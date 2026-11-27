import random
import string
from pathlib import Path

import pandas as pd

# ===============================================================
# ALGORITMO GENETICO:
# ===============================================================


# ==========================================
# CONFIGURACION DEL ALFABETO
# ==========================================

ALFABETO = string.ascii_uppercase + string.digits + " "


# ==========================================
# CREACION DE INDIVIDUOS Y POBLACION
# ==========================================

def crear_individuo(longitud, alfabeto):
    """Genera una cadena aleatoria."""
    return "".join(
        random.choice(alfabeto)
        for _ in range(longitud)
    )


def crear_poblacion(tamano, longitud, alfabeto):
    """Genera una poblacion de individuos aleatorios."""
    return [
        crear_individuo(longitud, alfabeto)
        for _ in range(tamano)
    ]


# ==========================================
# EVALUACION FITNESS
# ==========================================

def evaluar_fitness(individuo, objetivo):
    """Cuenta los caracteres que coinciden con el objetivo."""
    return sum(
        1
        for s, t in zip(individuo, objetivo)
        if s == t
    )


# ==========================================
# SELECCION POR TORNEO
# ==========================================

def seleccion_torneo(poblacion, fitness_poblacion, k=3):
    """Selecciona al mejor individuo de un torneo."""

    seleccionados = random.sample(
        range(len(poblacion)),
        min(k, len(poblacion))
    )

    mejor_indice = max(
        seleccionados,
        key=lambda x: fitness_poblacion[x]
    )

    return poblacion[mejor_indice]


# ==========================================
# CRUCE DE UN PUNTO
# ==========================================

def cruce_punto(padre1, padre2, prob_cruce=0.85):
    """Combina dos padres mediante un punto de corte."""

    if len(padre1) > 1 and random.random() < prob_cruce:

        punto_corte = random.randint(1, len(padre1) - 1)

        hijo1 = padre1[:punto_corte] + padre2[punto_corte:]
        hijo2 = padre2[:punto_corte] + padre1[punto_corte:]

        return hijo1, hijo2

    return padre1, padre2


# ==========================================
# MUTACION
# ==========================================

def mutacion(individuo, prob_mutacion, alfabeto):
    """Modifica caracteres aleatoriamente."""

    resultado = []

    for caracter in individuo:

        if random.random() < prob_mutacion:
            resultado.append(random.choice(alfabeto))
        else:
            resultado.append(caracter)

    return "".join(resultado)


# ==========================================
# ALGORITMO GENETIC PRINCIPAL
# ==========================================

def ejecutar_algoritmo_genetico(
    mensaje_oculto,
    tam_poblacion,
    max_generaciones,
    prob_cruce,
    prob_mutacion
):
    """Reconstruye el mensaje mediante un algoritmo genetico."""

    longitud = len(mensaje_oculto)

    # Incluir caracteres del mensaje objetivo
    alfabeto = "".join(
        dict.fromkeys(ALFABETO + mensaje_oculto)
    )

    # Crear poblacion inicial
    poblacion = crear_poblacion(
        tam_poblacion,
        longitud,
        alfabeto
    )

    historial_max = []

    mejor_individuo = ""
    mejor_fitness = -1
    generaciones_ejecutadas = 0

    # Ciclo evolutivo
    for generacion in range(max_generaciones):

        # Evaluar fitness de toda la poblacion
        fitness_poblacion = [
            evaluar_fitness(individuo, mensaje_oculto)
            for individuo in poblacion
        ]

        # Obtener el mejor individuo
        indice_mejor = fitness_poblacion.index(
            max(fitness_poblacion)
        )

        individuo_mejor = poblacion[indice_mejor]
        fitness_mejor = fitness_poblacion[indice_mejor]

        # Actualizar mejor solucion global
        if fitness_mejor > mejor_fitness:
            mejor_individuo = individuo_mejor
            mejor_fitness = fitness_mejor

        generaciones_ejecutadas = generacion + 1

        # Registrar historial
        historial_max.append(mejor_fitness)

        # Comprobar convergencia
        if mejor_fitness == longitud:
            break

        # Crear nueva poblacion conservando el mejor
        nueva_poblacion = [mejor_individuo]

        while len(nueva_poblacion) < tam_poblacion:

            # Seleccionar padres
            padre1 = seleccion_torneo(
                poblacion,
                fitness_poblacion
            )

            padre2 = seleccion_torneo(
                poblacion,
                fitness_poblacion
            )

            # Cruzar padres
            hijo1, hijo2 = cruce_punto(
                padre1,
                padre2,
                prob_cruce
            )

            # Aplicar mutacion
            hijo1 = mutacion(
                hijo1,
                prob_mutacion,
                alfabeto
            )

            hijo2 = mutacion(
                hijo2,
                prob_mutacion,
                alfabeto
            )

            nueva_poblacion.append(hijo1)

            if len(nueva_poblacion) < tam_poblacion:
                nueva_poblacion.append(hijo2)

        poblacion = nueva_poblacion

    # Devolver resultados
    return {
        "estado": "exitoso",
        "mensaje_oculto_objetivo": mensaje_oculto,
        "solucion_obtenida": mejor_individuo,
        "aptitud_alcanzada": mejor_fitness,
        "generaciones_ejecutadas": generaciones_ejecutadas,
        "convergencia": mejor_individuo == mensaje_oculto,
        "historial_fitness_max": historial_max
    }


