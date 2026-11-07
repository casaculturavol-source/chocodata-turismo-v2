# CHOCÓDATA · Versión sincronizada con el estudio turístico

Interfaz web para consultar el análisis multifuente, la evaluación de modelos y los pronósticos 2026–2027. Incorpora un modelo Random Forest persistido y el escenario de referencia estacional. La demostración pública no permite reentrenar ni activar modelos desde internet.

## Ejecución
```bash
python -m pip install -r requirements.txt
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Abrir `http://127.0.0.1:8000` y `http://127.0.0.1:8000/docs`.

## Subir a Render
Reemplaza los archivos de la **raíz** del repositorio GitHub `chocodata-render` con el contenido **descomprimido** de esta carpeta (`backend/`, `frontend/`, `data/`, `models/`, `requirements.txt`, `render.yaml`). Render realizará el despliegue automático en el servicio ya creado.

La interfaz muestra el indicador de llegadas aéreas como aproximación a la actividad turística, sin identificar el número total de turistas.
