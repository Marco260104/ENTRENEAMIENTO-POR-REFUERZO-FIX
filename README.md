# Laboratorio RL: CartPole, MountainCar e Inventario

**Estudiante:** Marco Torres  
**Carrera:** Ingeniería de Software – 7mo Semestre  
**Docente:** Isaac Torres  
**Materia:** Aprendizaje por Refuerzo 

---

## Estructura del proyecto

```
AprendizajeRefuerzo/
├── app.py                        # Servidor Flask (RL)
├── train_all.py                  # Entrena todos los agentes RL
├── requirements.txt              # Dependencias
│
├── src/
│   ├── parte_a.py                # CartPole — agente roto vs corregido
│   ├── parte_b.py                # MountainCar-v0
│   ├── parte_c.py                # Inventario (MDP propio)
│   └── utils.py                  # Utilidades compartidas
│
├── artifacts/
│   ├── models/                   # Q-tables entrenadas (.npz)
│   │   ├── cartpole_fixed.npz
│   │   ├── cartpole_broken.npz
│   │   ├── mountaincar.npz
│   │   └── inventario.npz
│   ├── figures/                  # Gráficos generados
│   │   ├── cartpole_roto_vs_corregido.png
│   │   ├── mountaincar_curva.png
│   │   └── inventario_curva.png
│   └── metrics/                  # Métricas de evaluación (.json)
│
├── templates/
│   └── index.html                # Interfaz web (3 pestañas)
├── static/
│   └── styles.css
└── venv/                         # Entorno virtual Python
```

---

## Instalación y ejecución

### 1. Crear y activar el entorno virtual

```powershell
python -m venv venv
.\venv\Scripts\activate
```

### 2. Instalar dependencias

```powershell
pip install -r requirements.txt
```

### 3. Ejecutar el servidor Flask

```powershell
python app.py
```

Abre el navegador en: **http://localhost:5000**

---

## Pestañas del servidor web

| Pestaña | Contenido |
|---------|-----------|
| **Parte A · CartPole** | Curva de aprendizaje, comparación roto vs corregido, predicción en vivo |
| **Parte B · MountainCar** | Q-table shape, hiperparámetros, predicción en vivo |
| **Parte C · Inventario** | MDP propio, curva de recompensa, predicción de pedido |

---

## Comandos de referencia rápida

```powershell
# Activar entorno
.\venv\Scripts\activate

# Re-entrenar agentes RL
python train_all.py

# Iniciar servidor
python app.py

# Desactivar entorno
deactivate
```
