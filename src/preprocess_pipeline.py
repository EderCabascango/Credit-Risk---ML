# ===============================================
# 📦 PIPELINE DE PREPROCESAMIENTO PARA CREDIT RISK
# ===============================================

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def build_preprocessor():
    """
    Construye y devuelve un pipeline de preprocesamiento para el dataset de crédito.
    Escala las variables numéricas y codifica las categóricas.
    """

    num_features = [
        "person_age", "person_income", "person_emp_length",
        "loan_amnt", "loan_int_rate", "loan_percent_income",
        "cb_person_cred_hist_length"
    ]

    cat_features = [
        "person_home_ownership", "loan_intent",
        "loan_grade", "cb_person_default_on_file"
    ]

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop="first", handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_features),
            ('cat', categorical_transformer, cat_features)
        ]
    )

    return preprocessor
