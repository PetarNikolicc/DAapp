import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import seaborn as sns
import matplotlib.pyplot as plt

# Anpassa sidans bredd och layout
st.set_page_config(page_title="Churn Prediction App", layout="wide")

# Styla appen med CSS
st.markdown(
    """
    <style>
    .main {
        background-color: #404040;
        padding: 20px;
    }
     h1 {
        color: #FAEBD7; /* Antikvit */
        text-align: center;
        font-size: 3em;
    }
    h2 {
        color: #FAEBD7;
        text-align: center;
        font-size: 2.2em;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True
)

# Titel och beskrivning
st.title("Kundavhoppsprediktion med Neural Nätverk")
st.write("En dynamisk applikation för att förutsäga kundavhopp med hjälp ett tränat neuralt nätverk.")

# Sidopanel för interaktiva hyperparametrar
st.sidebar.header("Justera hyperparametrar")
learning_rate = st.sidebar.slider("Lärandehastighet (learning rate)", 0.0001, 0.01, 0.001, step=0.0001)
neurons = st.sidebar.slider("Antal neuroner i första lagret", 4, 64, 8, step=4)
batch_size = st.sidebar.slider("Batch-storlek", 16, 128, 32, step=16)
epochs = st.sidebar.slider("Antal epoker", 10, 100, 50, step=10)

# Ladda upp CSV-filen
uploaded_file = st.file_uploader("Ladda upp din CSV-fil för kundavhopp", type="csv")

if uploaded_file is not None:
    # Läs in CSV-filen
    df = pd.read_csv(uploaded_file)

    # Visa de första raderna i datasetet
    st.write("### Förhandsgranskning av data")
    st.write(df.head())

    # Förbehandling av data
    st.write("### Förbehandling av data")
    
    # Progress bar för dataförbehandling
    st.write("Bearbetar data...")
    data_progress = st.progress(0)

    # Definiera features och target
    X = df[['CreditScore', 'Age', 'Complain', 'Balance']]
    y = df['Exited']

    # Dela upp data i tränings- och testset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Skala data
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Uppdatera progress bar till 100%
    data_progress.progress(100)

    # Dynamiska fakta om data
    st.write("### Fakta om datasetet")

    # Beräkna några relevanta fakta
    num_train = X_train.shape[0]
    num_test = X_test.shape[0]
    num_features = X_train.shape[1]
    churn_rate = y_train.mean() * 100
    train_test_ratio = num_train / num_test

    # Presentera dessa fakta med st.metric och kolumner för en bättre layout
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(label="Antal träningsdata", value=num_train)

    with col2:
        st.metric(label="Antal testdata", value=num_test)

    with col3:
        st.metric(label="Antal features", value=num_features)

    with col4:
        st.metric(label="Churn rate i träningsdata", value=f"{churn_rate:.2f} %")

    with col5:
        st.metric(label="Tränings/Test data ratio", value=f"{train_test_ratio:.2f}")

    # Förklaring av faktorerna
    st.write("""
    - **Antal träningsdata**: Antalet exempel som används för att träna modellen.
    - **Antal testdata**: Antalet exempel som används för att testa modellens prestanda.
    - **Antal features**: Antalet oberoende variabler (kolumner) i datasetet som används för att göra förutsägelser.
    - **Churn rate**: Andelen kunder som har churnat (avhoppat) i träningsdatan, uttryckt i procent.
    - **Tränings/Test ratio**: Förhållandet mellan tränings- och testdata.
    """)

    # **Lägg till grafen här**
    st.write("### Åldersfördelning för churnade och icke-churnade kunder")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df, x='Age', hue='Exited', multiple='stack', bins=30, palette='coolwarm', ax=ax)
    ax.set_title('Åldersfördelning för churnade och icke-churnade kunder')
    ax.set_xlabel('Ålder')
    ax.set_ylabel('Antal kunder')

    # Visa grafen i Streamlit
    st.pyplot(fig)

    # Skapa och träna neural nätverksmodell med optimerade hyperparametrar
    st.write("### Träning av Neural Nätverk med anpassade parametrar")

    def create_model(learning_rate, neurons):
        model = Sequential()
        model.add(Dense(neurons, input_dim=X_train_scaled.shape[1], activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        return model

    # Träna modellen med användarens valda parametrar
    model = create_model(learning_rate=learning_rate, neurons=neurons)
    
    # Progress bar för träning
    st.write("Tränar modellen...")
    train_progress = st.progress(0)
    
    for epoch in range(epochs):
        model.fit(X_train_scaled, y_train, epochs=1, batch_size=batch_size, verbose=0)
        train_progress.progress((epoch + 1) / epochs)

    st.write("Modellen har tränats färdigt.")

    # Utvärdera modellen
    st.write("### Utvärdering av modellen")
    
    # Gör prediktioner
    y_pred = (model.predict(X_test_scaled) > 0.5).astype(int)

    # Beräkna nyckelmetrik
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    # Dynamiska fakta om modellens prestanda
    st.write("### Modellens prestanda")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label="Precision", value=f"{precision:.2f}")
    col2.metric(label="Recall", value=f"{recall:.2f}")
    col3.metric(label="F1-score", value=f"{f1:.2f}")
    col4.metric(label="ROC AUC", value=f"{roc_auc:.2f}")

    # Visualisera Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    st.write("### Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)

    # Visualisera ROC-kurva
    fpr, tpr, _ = roc_curve(y_test, model.predict(X_test_scaled))
    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.2f}')
    ax_roc.plot([0, 1], [0, 1], linestyle='--')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('ROC Curve')
    ax_roc.legend()
    st.pyplot(fig_roc)

    st.write("### Slutsats")
    st.write("""
    Med dessa resultat kan vi dra slutsatsen att modellen är mycket effektiv på att förutsäga kundavhopp.
    Företaget kan använda denna modell för att proaktivt identifiera kunder som riskerar att avsluta sina tjänster,
    vilket kan leda till förbättrad kundretention och minskade förluster.
    """)
else:
    st.write("Vänligen ladda upp en CSV-fil för att fortsätta.")
