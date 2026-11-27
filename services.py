import random
import string
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from fastapi import HTTPException

# ===============================================================
#K-MEANS
# ===============================================================

SEED_KMEANS = 20


def inicializacion_centroides(X, K):
    """Selecciona K puntos aleatorios como centroides iniciales."""
    indices_aleatorios = np.random.choice(
        X.shape[0], K, replace=False
    )
    centroides = X[indices_aleatorios].copy()
    return centroides


def asignar_cluster(X, centroides):
    """Asigna cada punto al centroide mas cercano."""
    distancias = np.linalg.norm(
        X[:, np.newaxis, :] - centroides[np.newaxis, :, :],
        axis=2
    )
    indices_cluster = np.argmin(distancias, axis=1)
    return indices_cluster


def recalcular_centroides(X, etiquetas, K):
    """Calcula nuevamente los centroides de cada cluster."""
    N, D = X.shape
    nuevos_centroides = np.zeros((K, D))

    for k in range(K):
        puntos_cluster = X[etiquetas == k]

        if len(puntos_cluster) > 0:
            nuevos_centroides[k] = np.mean(
                puntos_cluster, axis=0
            )
        else:
            nuevos_centroides[k] = X[
                np.random.choice(N)
            ]

    return nuevos_centroides


def calcular_inercia(X, centroides, etiquetas):
    """Calcula la inercia WCSS."""
    inercia = 0

    for k in range(len(centroides)):
        puntos_cluster = X[etiquetas == k]

        if len(puntos_cluster) > 0:
            inercia += np.sum(
                (puntos_cluster - centroides[k]) ** 2
            )

    return float(inercia)


def ejecutar_kmeans(k_clusters, max_iteraciones, datos):
    """Ejecuta el algoritmo K-Means y devuelve sus resultados."""

    X = np.asarray(datos, dtype=float)

    # Validaciones de los datos
    if X.ndim != 2 or X.shape[0] < k_clusters:
        raise HTTPException(
            status_code=400,
            detail="Se necesitan al menos K muestras para formar los clusters."
        )

    if X.shape[1] == 0:
        raise HTTPException(
            status_code=400,
            detail="Cada muestra debe contener al menos una característica."
        )

    if not np.isfinite(X).all():
        raise HTTPException(
            status_code=400,
            detail="Los datos deben contener únicamente valores numéricos finitos."
        )

    # Inicialización reproducible
    generador = np.random.default_rng(SEED_KMEANS)
    indices = generador.choice(
        X.shape[0], k_clusters, replace=False
    )
    centroides = X[indices].copy()

    # Iteraciones del algoritmo
    for iteracion in range(max_iteraciones):

        etiquetas = asignar_cluster(X, centroides)

        nuevos_centroides = recalcular_centroides(
            X, etiquetas, k_clusters
        )

        desplazamiento = np.linalg.norm(
            nuevos_centroides - centroides
        )

        centroides = nuevos_centroides

        if desplazamiento <= 1e-4:
            break

    # Asignación final usando los centroides definitivos
    etiquetas = asignar_cluster(X, centroides)

    # Cálculo de la inercia
    inercia = calcular_inercia(
        X, centroides, etiquetas
    )

    return {
        "estado": "exitoso",
        "k_utilizado": k_clusters,
        "num_muestras": int(X.shape[0]),
        "centroides_finales": centroides.tolist(),
        "etiquetas_asignadas": etiquetas.tolist(),
        "inercia_wcss": inercia
    }

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


