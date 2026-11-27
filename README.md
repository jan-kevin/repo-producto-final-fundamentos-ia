# Producto Final Fundamentos de IA - API RESTful de Inteligencia Artificial

API REST desarrollada con Python y FastAPI que permite ejecutar tres algoritmos de Inteligencia Artificial: un algoritmo genético, un árbol de decisión y el algoritmo de agrupamiento K-Means.

El proyecto fue desarrollado como actividad final de la asignatura **Fundamentos de Inteligencia Artificial** por los alumnos:
**- Langevoort Gonzalez Jan Kevin**
**- Cervantes Salgado Daniel**
**- Hernandez Leyva Moises**

## Tecnologías utilizadas

- Python
- FastAPI
- Uvicorn
- Pydantic
- NumPy
- Pandas
- Scikit-learn
- OpenPyXL

## Estructura del proyecto

```text
producto-final/
│
├── main.py
├── models.py
├── services.py
├── requirements.txt
├── README.md
├── dataset_cardiaco_simulado.xlsx
└── .gitignore
```

### Descripción de los archivos

| Archivo | Descripción |
|---|---|
| `main.py` | Punto de entrada de FastAPI y definición de los endpoints. |
| `models.py` | Modelos Pydantic para validar las solicitudes y respuestas. |
| `services.py` | Implementación de los algoritmos y su lógica de ejecución. |
| `requirements.txt` | Dependencias necesarias para ejecutar el proyecto. |
| `dataset_cardiaco_simulado.xlsx` | Dataset utilizado para entrenar el árbol de decisión. |
| `README.md` | Documentación del proyecto. |

## Instalación y configuración

### 1. Clonar el repositorio

```bash
git clone URL_DEL_REPOSITORIO
cd producto-final
```

### 2. Crear un entorno virtual

```bash
python -m venv .venv
```

### 3. Activar el entorno virtual

En Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Instalar las dependencias

```bash
pip install -r requirements.txt
```

## Ejecución del proyecto

Para iniciar el servidor de desarrollo, ejecutar:

```bash
uvicorn main:app --reload
```

Una vez iniciado, la API estará disponible en:

- **API:** http://127.0.0.1:8000
- **Swagger UI:** http://127.0.0.1:8000/docs
- **ReDoc:** http://127.0.0.1:8000/redoc

La interfaz Swagger UI permite probar los tres endpoints directamente desde el navegador.

---

## Endpoints disponibles

### 1. Algoritmo genético

**Método:** `POST`

**Ruta:** `/genetico`

Implementa un algoritmo genético que busca reconstruir un mensaje oculto mediante una población de individuos, selección por torneo, cruce, mutación y elitismo.

**Parámetros de entrada:**

| Parámetro | Tipo | Descripción |
|---|---|---|
| `mensaje_oculto` | string | Mensaje que el algoritmo debe encontrar. |
| `tam_poblacion` | integer | Cantidad de individuos de la población (11–500). |
| `max_generaciones` | integer | Número máximo de generaciones (2–1000). |
| `prob_cruce` | float | Probabilidad de cruce (0–1). |
| `prob_mutacion` | float | Probabilidad de mutación (0–0.5). |

**Ejemplo de solicitud:**

```json
{
  "mensaje_oculto": "IA",
  "tam_poblacion": 100,
  "max_generaciones": 100,
  "prob_cruce": 0.8,
  "prob_mutacion": 0.1
}
```

**Respuesta esperada:**

```json
{
  "estado": "exitoso",
  "mensaje_oculto_objetivo": "IA",
  "solucion_obtenida": "IA",
  "aptitud_alcanzada": 2,
  "generaciones_ejecutadas": 3,
  "convergencia": true,
  "historial_fitness_max": [1, 1, 2]
}
```

*Nota: los resultados y el número de generaciones pueden variar debido al carácter aleatorio del algoritmo.*

### 2. Árbol de decisión

**Método:** `POST`

**Ruta:** `/arbol_decision`

Implementa un modelo de aprendizaje supervisado mediante un árbol de decisión para estimar el riesgo asociado con cardiopatía a partir de características de un paciente.

El modelo utiliza el dataset `dataset_cardiaco_simulado.xlsx` para su entrenamiento.

**Características utilizadas:**

- Edad.
- Presión sistólica.
- Colesterol.
- Frecuencia cardíaca máxima.
- Angina inducida por ejercicio.

**Parámetros de entrada:**

| Parámetro | Tipo | Rango |
|---|---|---|
| `edad` | integer | 18–100 |
| `presion_sistolica` | integer | 80–220 |
| `colesterol` | integer | 100–500 |
| `frecuencia_max` | integer | 60–220 |
| `angina_ejercicio` | integer | 0 o 1 |

**Ejemplo de solicitud:**

```json
{
  "edad": 55,
  "presion_sistolica": 140,
  "colesterol": 250,
  "frecuencia_max": 150,
  "angina_ejercicio": 1
}
```

**Respuesta esperada:**

```json
{
  "estado": "exitoso",
  "paciente_evaluado": {
    "edad": 55,
    "presion_sistolica": 140,
    "colesterol": 250,
    "frecuencia_max": 150,
    "angina_ejercicio": 1
  },
  "clase_predicha": 1,
  "diagnostico_texto": "El modelo identifica caracteristicas asociadas con cardiopatia.",
  "probabilidad_riesgo": 1.0,
  "nivel_riesgo": "Alto"
}
```

**Nota importante:** El dataset utilizado es simulado y el modelo tiene fines exclusivamente educativos. Sus resultados no representan diagnósticos médicos ni deben utilizarse para tomar decisiones clínicas.

### 3. Algoritmo K-Means

**Método:** `POST`

**Ruta:** `/cluster`

Implementa el algoritmo de aprendizaje no supervisado K-Means para agrupar datos según su proximidad a determinados centroides.

El algoritmo realiza la inicialización de centroides, asignación de muestras al centroide más cercano, actualización de centroides y cálculo de la inercia WCSS.

**Parámetros de entrada:**

| Parámetro | Tipo | Descripción |
|---|---|---|
| `k_clusters` | integer | Número de grupos que se desean generar (2–10). |
| `max_iteraciones` | integer | Número máximo de iteraciones (6–500). |
| `datos` | array | Conjunto de muestras numéricas que se desean agrupar. |

**Ejemplo de solicitud:**

```json
{
  "k_clusters": 3,
  "max_iteraciones": 100,
  "datos": [
    [2.0, 3.0],
    [2.5, 3.5],
    [3.0, 2.0],
    [8.0, 9.0],
    [8.5, 8.0],
    [9.0, 8.5],
    [14.0, 2.0],
    [14.5, 3.0],
    [15.0, 2.5]
  ]
}
```

**Respuesta esperada:**

```json
{
  "estado": "exitoso",
  "k_utilizado": 3,
  "num_muestras": 9,
  "centroides_finales": [
    [8.5, 8.5],
    [14.5, 2.5],
    [2.5, 2.8333333333333335]
  ],
  "etiquetas_asignadas": [2, 2, 2, 0, 0, 0, 1, 1, 1],
  "inercia_wcss": 3.6666666666666665
}
```

*Nota: las etiquetas identifican el grupo asignado a cada muestra. Los números de los grupos no representan un orden de importancia.*

---

## Validaciones

La API utiliza **Pydantic** para validar los datos recibidos en cada solicitud.

Si los parámetros no cumplen con los tipos o rangos establecidos, FastAPI devuelve una respuesta HTTP `422 Unprocessable Entity`.

Además, K-Means verifica que existan suficientes muestras para generar los grupos solicitados y que los datos sean numéricos y finitos.

## Pruebas

Los tres endpoints fueron ejecutados y probados mediante Swagger UI:

1. `/genetico`: reconstrucción del mensaje oculto.
2. `/arbol_decision`: predicción de riesgo mediante el árbol de decisión.
3. `/cluster`: agrupamiento de muestras mediante K-Means.