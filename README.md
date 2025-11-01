# 🧠 Credit Risk Prediction  
*Predicción de incumplimiento crediticio usando Machine Learning*

---

## 📘 Descripción general

Este proyecto tiene como objetivo **predecir la probabilidad de incumplimiento (default)** en préstamos personales a partir de información socioeconómica y financiera de los solicitantes.  
Utiliza técnicas de **análisis exploratorio de datos (EDA)**, **limpieza**, **ingeniería de variables** y **modelos de clasificación supervisada** para identificar patrones asociados al riesgo crediticio.

---

## 🧩 Dataset

El dataset contiene información de más de **30,000 registros**, cada uno representando un cliente con las siguientes variables principales:

| Tipo | Variable | Descripción |
|------|-----------|-------------|
| Numérica | `person_age` | Edad del solicitante |
| Numérica | `person_income` | Ingreso anual o mensual del cliente |
| Numérica | `person_emp_length` | Años en el empleo actual |
| Categórica | `person_home_ownership` | Tipo de vivienda (RENT, OWN, MORTGAGE) |
| Categórica | `loan_intent` | Propósito del préstamo |
| Categórica | `loan_grade` | Calificación crediticia |
| Numérica | `loan_amnt` | Monto solicitado |
| Numérica | `loan_int_rate` | Tasa de interés anual |
| Numérica | `loan_percent_income` | Proporción préstamo/ingreso |
| Categórica | `cb_person_default_on_file` | Historial de impago previo |
| Numérica | `cb_person_cred_hist_length` | Antigüedad del historial crediticio |
| Target | `loan_status` | 0 = pagado, 1 = default |

---

## 🔍 EDA (Análisis Exploratorio de Datos)

Durante el EDA se realizaron los siguientes pasos:

1. **Inspección inicial:** detección de nulos, tipos de datos y outliers.  
2. **Distribuciones univariadas y multivariadas.**  
3. **Análisis de relación con la variable objetivo (`loan_status`).**  
4. **Evaluación de correlaciones numéricas y dependencias categóricas (chi-cuadrado).**

**Principales hallazgos:**
- Ingresos más bajos y tasas de interés altas están asociados con mayor riesgo.  
- Los clientes que alquilan (`RENT`) o tienen préstamos para consolidar deudas presentan más defaults.  
- `loan_int_rate` y `loan_grade` tienen una correlación de 0.94 → indican posible colinealidad.  
- Las variables financieras (`income`, `percent_income`, `loan_grade`) son los predictores más influyentes.

---

## ⚙️ Preprocesamiento

- Limpieza de outliers y valores imposibles.  
- Winsorización para limitar extremos.  
- Codificación de variables categóricas (OneHotEncoder).  
- Escalado de variables numéricas (StandardScaler).  
- División del dataset en **train/test (80/20)** con estratificación.  
- Balanceo de clases con **SMOTE** para mitigar el desbalance entre defaults y no-defaults.

---

## 🤖 Modelos aplicados

Los siguientes algoritmos serán evaluados:

- **Regresión Logística** → baseline interpretable.  
- **Random Forest** → modelo no lineal robusto.  
- **LightGBM / XGBoost** → optimización basada en boosting.  
- (Opcional) **Red Neuronal simple (MLP)** para comparación.

---

## 📊 Métricas de evaluación

| Métrica | Descripción |
|----------|-------------|
| **Precision** | Exactitud de las predicciones positivas |
| **Recall (Sensibilidad)** | Qué tan bien identifica los defaults reales |
| **F1-Score** | Balance entre precisión y recall |
| **ROC-AUC** | Capacidad global de discriminación del modelo |

---

## 🧠 Resultados esperados

- Determinar las variables más influyentes en el riesgo crediticio.  
- Construir un modelo con **alto recall en la clase 1 (default)** para minimizar pérdidas financieras.  
- Implementar un pipeline reproducible para evaluación y mejora continua.

---

## 🛠️ Tecnologías utilizadas

- **Python 3.13**
- **Pandas, NumPy, Matplotlib, Seaborn**
- **Scikit-learn**
- **Imbalanced-learn (SMOTE)**
- **LightGBM / XGBoost**

---

## 🧾 Estructura del proyecto

