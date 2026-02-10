# 🔥 Grid-Based Forest Fire Early Warning System

A production-style academic project that predicts localized forest fire risk using Machine Learning and real-time weather/satellite data.

![Technology Stack](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Framework](https://img.shields.io/badge/FastAPI-0.109-green?logo=fastapi)
![ML](https://img.shields.io/badge/scikit--learn-1.4-orange?logo=scikit-learn)
![Map](https://img.shields.io/badge/Leaflet.js-1.9-brightgreen?logo=leaflet)

## 🎯 Project Objective

Build a real-world decision-support system that:
- **Divides forest regions into spatial grid cells** (12 grids per region)
- **Predicts fire risk independently for each grid** using Random Forest ML
- **Visualizes risk on an interactive map** with color-coded grids
- **Enables management teams** to identify exact high-risk locations

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         FRONTEND                                 │
│  ┌─────────────┐  ┌───────────────┐  ┌───────────────────────┐  │
│  │   Region    │  │   Leaflet     │  │    Explanation        │  │
│  │  Selector   │  │     Map       │  │      Panel            │  │
│  └─────────────┘  └───────────────┘  └───────────────────────┘  │
└──────────────────────────┬──────────────────────────────────────┘
                           │ REST API
┌──────────────────────────▼──────────────────────────────────────┐
│                     BACKEND (FastAPI)                            │
│  ┌─────────────┐  ┌───────────────┐  ┌───────────────────────┐  │
│  │    Grid     │  │   Weather     │  │ ML Model (Random      │  │
│  │   Manager   │  │   Fetcher     │  │ Forest Classifier)    │  │
│  └─────────────┘  └───────────────┘  └───────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
hack2/
├── data/
│   └── forest_fires.csv           # Historical wildfire dataset
├── models/
│   ├── fire_risk_model.pkl        # Trained Random Forest model
│   ├── scaler.pkl                 # Feature scaler
│   └── model_metadata.json        # Model metrics and info
├── backend/
│   ├── main.py                    # FastAPI application
│   ├── model.py                   # ML model wrapper
│   ├── grid.py                    # Spatial grid logic
│   ├── weather.py                 # Weather API integration
│   └── requirements.txt           # Python dependencies
├── frontend/
│   ├── index.html                 # Main UI
│   ├── styles.css                 # Dark theme styling
│   └── app.js                     # Map & interaction logic
└── notebooks/
    └── model_training.py          # Model training script
```

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
cd hack2/backend
pip install -r requirements.txt
```

### Step 2: Train the ML Model

```bash
cd hack2
python notebooks/model_training.py
```

### Step 3: Start the Backend Server

```bash
cd hack2/backend
uvicorn main:app --reload --port 8000
```

### Step 4: Open the Frontend

```bash
cd hack2/frontend
python -m http.server 3000
```

Open http://localhost:3000 in your browser.

## 📊 Model Evaluation

The Random Forest classifier is trained on the Algerian Forest Fires dataset with these results:

| Metric | Score |
|--------|-------|
| Accuracy | ~85% |
| Precision | ~84% |
| Recall | ~87% |
| F1 Score | ~85% |

### Feature Importance

| Feature | Description | Importance |
|---------|-------------|------------|
| Temperature | Air temperature (°C) | High |
| FFMC | Fine Fuel Moisture Code | High |
| DMC | Duff Moisture Code | Medium |
| RH | Relative Humidity (%) | Medium |
| ISI | Initial Spread Index | Medium |
| Ws | Wind Speed (km/h) | Medium |
| FWI | Fire Weather Index | Low-Medium |

## 🗺️ Grid-Based Design

Each forest region is divided into a **4×3 grid (12 cells)**:

```
┌──────┬──────┬──────┬──────┐
│ 0,0  │ 0,1  │ 0,2  │ 0,3  │
├──────┼──────┼──────┼──────┤
│ 1,0  │ 1,1  │ 1,2  │ 1,3  │
├──────┼──────┼──────┼──────┤
│ 2,0  │ 2,1  │ 2,2  │ 2,3  │
└──────┴──────┴──────┴──────┘
```

**Academic Justification:**
> "This grid-based approach enables spatially localized fire risk modeling for targeted early intervention. By dividing forest regions into discrete cells, emergency responders can identify exact high-risk locations rather than responding to forest-wide alerts, enabling more efficient resource allocation and faster response times."

## 🎨 Risk Visualization

| Risk Score | Category | Color | Recommended Action |
|------------|----------|-------|-------------------|
| 0-33 | Low | 🟢 Green | Standard monitoring |
| 34-66 | Medium | 🟡 Yellow | Increase patrols |
| 67-100 | High | 🔴 Red | Deploy fire prevention |

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/regions` | List available forest regions |
| GET | `/api/grids/{region}` | Get grid cells for a region |
| POST | `/api/predict` | Predict fire risk for all grids |
| GET | `/api/grid/{region}/{grid_id}` | Get detailed explanation |

## 🎓 Viva Q&A

**Q: Why use grid-based prediction instead of whole-forest prediction?**
> Grid-based prediction allows localized risk assessment, enabling fire management teams to pinpoint exact high-risk areas and deploy resources efficiently.

**Q: Why Random Forest over Deep Learning?**
> Random Forest is interpretable, works well with tabular data, handles missing values gracefully, and provides feature importance for explainability—crucial for decision-making systems.

**Q: How does real-time data improve predictions?**
> Weather conditions like temperature, humidity, and wind speed change rapidly. Real-time data ensures predictions reflect current conditions, not historical averages.

**Q: What makes this system practical?**
> 1. Localized predictions for targeted response
> 2. Explainable AI for transparent decision-making
> 3. Real-time data for current conditions
> 4. Interactive visualization for quick understanding

## 🏆 Hackathon Demo Flow

1. **Introduction** (30 sec): "This is a Grid-Based Forest Fire Early Warning System..."
2. **Select Region** (10 sec): Choose California from dropdown
3. **Predict Risk** (20 sec): Click predict, watch grids appear with colors
4. **Explore Grids** (30 sec): Click high-risk grid, show explanation panel
5. **Technical Overview** (30 sec): Explain ML model and architecture

## 📝 License

MIT License - Academic/Educational Use

---

**Built with ❤️ for Forest Fire Prevention**
