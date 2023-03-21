import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def main():
    st.title("Proyección de cantidad de alumnos")

    st.write("""
    Introduce la información de inscripciones y alumnos que no salen de la generación por semestre para predecir la cantidad de alumnos en el futuro.
    """)

    # Entrada de datos
    st.sidebar.title("Datos de inscripciones y alumnos que no salen por semestre")
    n_semesters = st.sidebar.number_input("Número de semestres:", min_value=1, value=5)
    inscriptions = st.sidebar.text_input("Inscripciones por semestre (separadas por comas):", value="100,120,130,110,125")
    not_leaving = st.sidebar.text_input("Alumnos que no salen por semestre (separadas por comas):", value="5,8,10,6,7")

    # Procesamiento de datos
    inscriptions_list = list(map(int, inscriptions.split(',')))
    not_leaving_list = list(map(int, not_leaving.split(',')))

    if len(inscriptions_list) != n_semesters or len(not_leaving_list) != n_semesters:
        st.sidebar.error("Por favor, proporciona la misma cantidad de valores que el número de semestres.")
    else:
        data = pd.DataFrame({
            'semester': np.arange(1, n_semesters + 1),
            'inscriptions': inscriptions_list,
            'not_leaving': not_leaving_list
        })

        st.subheader("Datos proporcionados")
        st.dataframe(data)

        # Entrenar modelo de regresión lineal
        X = data[['semester']]
        y = data['inscriptions'] - data['not_leaving']
        model = LinearRegression()
        model.fit(X, y)

        # Proyectar cantidad de alumnos en el futuro
        st.sidebar.title("Proyección")
        years = st.sidebar.number_input("Número de años a proyectar:", min_value=1, value=5)
        future_semesters = 2 * years

        future_data = pd.DataFrame({
            'semester': np.arange(n_semesters + 1, n_semesters + future_semesters + 1)
        })

        future_data['predicted_inscriptions'] = model.predict(future_data[['semester']])
        future_data['predicted_inscriptions'] = future_data['predicted_inscriptions'].clip(lower=0).round()
        future_students = int(future_data['predicted_inscriptions'].sum())

        st.subheader(f"Proyección de alumnos en {years} años")
        st.write(f"Se espera tener aproximadamente {future_students} alumnos en {years} años.")

if __name__ == "__main__":
    main()
