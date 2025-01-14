import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import umap.umap_ as umap
from scipy import stats
import joblib
import io
from sklearn import datasets
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc  
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objs as go

# -----------------------------------------------------------------------------------------------------------------------------
# Definiera save_dataset-funktionen 
def save_dataset(data):
    csv = data.to_csv(index=False)
    csv_bytes = io.BytesIO(csv.encode())
    st.download_button(
        label="Ladda ner bearbetat dataset",
        data=csv_bytes,
        file_name="bearbetat_dataset.csv",
        mime="text/csv"
    )

# ----------------------------------------------------------------------------------------------------------------------------
# Modul 1: Datainsamling 
def load_data():
    st.title("Datainsamling")

    # Val av dataset att ladda in
    dataset_option = st.sidebar.selectbox("Välj ett exempeldata eller ladda upp din egen fil", 
                                          ["Ladda upp dataset", "Iris", "Titanic"], key="dataset_selector")

    if dataset_option == "Ladda upp dataset":
        file = st.sidebar.file_uploader("Ladda upp din dataset (CSV/Excel)", type=["csv", "xlsx"], key="file_uploader")
        if file:
            try:
                if file.name.endswith(".csv"):
                    data = pd.read_csv(file)
                elif file.name.endswith(".xlsx"):
                    data = pd.read_excel(file)

                # Visa  dataframe head
                st.write("### Dataframe")
                st.dataframe(data.head(50))  

                return data

            except Exception as e:
                st.error(f"Ett fel uppstod vid uppladdning: {e}")
        else:
            st.info("Vänligen ladda upp en CSV- eller Excel-fil.")
    
    elif dataset_option == "Iris":
        data = sns.load_dataset("iris")
        st.success("Iris-datasetet laddades framgångsrikt!")
        st.write("Här är de första raderna av Iris-datasetet:")
        st.dataframe(data.head(50)) 

        return data

    elif dataset_option == "Titanic":
        data = sns.load_dataset("titanic")
        st.success("Titanic-datasetet laddades framgångsrikt!")
        st.write("Här är de första raderna av Titanic-datasetet:")
        st.dataframe(data.head(50)) 

        return data

    return None


# ----------------------------------------------------------------------------------------------------------------------------
# Modul 2: Dataförståelse och EDA (med beskrivningar)
def eda_module(data):
    st.title("Dataförståelse och EDA - Dashboard")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Datatyper")
        st.write(data.dtypes)

    with col2:
        st.subheader("Antal rader och kolumner")
        st.write(f"Antal rader: {data.shape[0]}, Antal kolumner: {data.shape[1]}")

    with col3:
        st.subheader("Saknade värden")
        st.write(data.isnull().sum())

    st.subheader("Statistisk beskrivning")
    st.write(data.describe())  # Flyttar beskrivningarna till EDA-modulen

    st.subheader("Korrelationsmatris")
    numeric_data = data.select_dtypes(include=["float64", "int64"])
    
    if numeric_data.empty:
        st.warning("Ditt dataset innehåller inga numeriska kolumner att beräkna korrelation för.")
    else:
        numeric_data = numeric_data.fillna(0)
        corr = numeric_data.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', cbar=True)
        st.pyplot(plt.gcf())

    st.subheader("Histogram")
    num_cols = data.select_dtypes(include=['float64', 'int64']).columns
    if len(num_cols) > 0:
        data[num_cols].hist(figsize=(10, 10), bins=20)
        st.pyplot(plt.gcf())

    save_dataset(data)

# ----------------------------------------------------------------------------------------------------------------------------
# Modul 3: Data Preprocessing med realtidsuppdatering av dataframe och statistiska tester
def preprocessing_module(data):
    st.title("Data Preprocessing")

    # Initiera historik för preprocessing om den inte redan finns
    if "preprocessing_history" not in st.session_state:
        st.session_state["preprocessing_history"] = [data.copy()]

    # Funktion för att ångra senaste ändring
    if st.sidebar.button("Ångra senaste ändring"):
        if len(st.session_state["preprocessing_history"]) > 1:
            st.session_state["preprocessing_history"].pop()
            data = st.session_state["preprocessing_history"][-1].copy()
            st.success("Senaste ändring ångrad!")
        else:
            st.warning("Det finns inga tidigare ändringar att ångra.")
    else:
        data = st.session_state["preprocessing_history"][-1]  # Sätt senaste data som aktiv

    # Skapa layout med 3 kolumner bredvid varandra
    col1, col2, col3 = st.columns(3)

    # Hantering av saknade värden
    with col1:
        st.subheader("Hantera saknade värden")
        missing_option = st.selectbox("Välj metod", 
                                      ["Ta bort rader med saknade värden", 
                                       "Fyll i saknade värden med median", 
                                       "Fyll i saknade värden med medelvärde", 
                                       "Fyll i saknade värden med ett specifikt värde"], 
                                      key="missing_option")
        if st.button("Bekräfta saknade värden"):
            if missing_option == "Ta bort rader med saknade värden":
                data = data.dropna()
            elif missing_option == "Fyll i saknade värden med median":
                data = data.fillna(data.median())
            elif missing_option == "Fyll i saknade värden med medelvärde":
                data = data.fillna(data.mean())
            
            st.session_state["preprocessing_history"].append(data.copy())  # Spara historik
            st.session_state["data"] = data  # Uppdatera session_state med den nya datan

    # Ta bort kolumner
    with col2:
        st.subheader("Ta bort kolumner")
        columns_to_drop = st.multiselect("Välj kolumner att ta bort", data.columns.tolist(), key="drop_columns")
        if st.button("Bekräfta borttagning av kolumner"):
            if len(columns_to_drop) > 0:
                data = data.drop(columns=columns_to_drop)
                st.session_state["preprocessing_history"].append(data.copy())  # Spara historik
                st.session_state["data"] = data  # Uppdatera session_state med den nya datan

    # Skala data
    with col3:
        st.subheader("Skala data")
        scale_option = st.selectbox("Välj skalningsmetod", ["StandardScaler", "MinMaxScaler"], key="scale_option")
        num_cols = st.multiselect("Välj numeriska kolumner att skala", data.select_dtypes(include=['float64', 'int64']).columns.tolist(), key="scaling_columns")
        if st.button("Bekräfta skalning"):
            if len(num_cols) > 0:
                if scale_option == "StandardScaler":
                    scaler = StandardScaler()
                elif scale_option == "MinMaxScaler":
                    scaler = MinMaxScaler()
                data[num_cols] = scaler.fit_transform(data[num_cols])
                st.session_state["preprocessing_history"].append(data.copy())  
                st.session_state["data"] = data  

    # Nästa rad med tre nya kolumner
    col4, col5, col6 = st.columns(3)

    # Statistiska tester - Shapiro-Wilk och Kruskal-Wallis
    with col4:
        st.subheader("Shapiro-Wilk test")
        selected_num_cols = st.multiselect("Välj kolumner för Shapiro-Wilk", data.select_dtypes(include=['float64', 'int64']).columns.tolist(), key="shapiro_columns")
        if st.button("Kör Shapiro-Wilk test"):
            for col in selected_num_cols:
                stat, p_value = stats.shapiro(data[col])
                st.write(f"Shapiro-Wilk test för {col}: p-värde = {p_value:.4f}")
                if p_value < 0.05:
                    st.write(f"{col} är inte normalfördelad.")
                else:
                    st.write(f"{col} är normalfördelad.")

    with col5:
        st.subheader("Kruskal-Wallis test")
        selected_num_cols_kruskal = st.multiselect("Välj kolumner för Kruskal-Wallis", data.select_dtypes(include=['float64', 'int64']).columns.tolist(), key="kruskal_columns")
        groups_col = st.selectbox("Välj en kolumn för grupper", data.columns.tolist(), key="kruskal_group_column")
        if st.button("Kör Kruskal-Wallis test"):
            if len(selected_num_cols_kruskal) > 0 and groups_col:
                groups = [data[data[groups_col] == g][selected_num_cols_kruskal[0]] for g in data[groups_col].unique()]
                stat, p_value = stats.kruskal(*groups)
                st.write(f"Kruskal-Wallis test för {selected_num_cols_kruskal[0]} grupperad efter {groups_col}: p-värde = {p_value:.4f}")

    # Visa uppdaterad data efter varje förändring
    st.subheader("Uppdaterad Data (efter bearbetning)")
    st.dataframe(data.head())  # Visar de första raderna av den bearbetade datan

    # Ladda ner bearbetad data
    with col6:
        st.subheader("Spara dataset")
        save_dataset(data)  # Möjlighet att spara datasetet efter bearbetning

    return data
# ----------------------------------------------------------------------------------------------------------------------------
# Modul 4: Feature Engineering

def feature_engineering_module(data):
    st.title("Feature Engineering")

    if st.checkbox("Lägg till nya variabler", key="feature_eng"):
        st.write("Skapa nya variabler baserat på befintliga kolumner")
        selected_columns = st.multiselect("Välj kolumner att skapa nya variabler från", data.columns.tolist(), key="select_columns")
        transformation = st.selectbox("Välj transformation", ["Summa", "Produkt", "Differens", "Medelvärde"], key="transformation_type")
        new_column_name = st.text_input("Ange namn för den nya variabeln", key="new_column_name")

        if st.button("Skapa ny variabel", key="create_new_variable"):
            if len(selected_columns) > 0 and new_column_name != "":
                if transformation == "Summa":
                    data[new_column_name] = data[selected_columns].sum(axis=1)
                elif transformation == "Produkt":
                    data[new_column_name] = data[selected_columns].prod(axis=1)
                elif transformation == "Differens" and len(selected_columns) == 2:
                    data[new_column_name] = data[selected_columns[0]] - data[selected_columns[1]]
                elif transformation == "Medelvärde":
                    data[new_column_name] = data[selected_columns].mean(axis=1)

                st.success(f"Ny variabel '{new_column_name}' har skapats!")
                st.dataframe(data.head())

    save_dataset(data)
    return data

# ----------------------------------------------------------------------------------------------------------------------------
# Modul 5: Dimensionalitetsreduktion (PCA/UMAP)
def method_test(data):
    st.write("Metodtest: PCA eller UMAP?")
    
    num_cols = data.select_dtypes(include=['float64', 'int64']).shape[1]
    num_rows = data.shape[0]

    if num_cols >= 10:
        st.write("PCA kan vara mer lämpligt för att reducera dimensionaliteten.")
    elif num_rows > 5000:
        st.write("UMAP är bättre på att hitta klustrar.")
    else:
        st.write("Båda metoderna kan fungera bra.")

def dimensionality_reduction_module(data):
    st.title("Dimensionalitetsreduktion med PCA och UMAP")
    method_test(data)

    reduction_method = st.selectbox("Välj reduktionsteknik", ["PCA", "UMAP", "PCA först, sedan UMAP"], key="reduction_method")
    num_cols = st.multiselect("Välj numeriska kolumner för reduktion", data.select_dtypes(include=['float64', 'int64']).columns.tolist(), key="dim_reduction_columns")
    
    if len(num_cols) > 0:
        max_components = min(len(num_cols), 10)
        if reduction_method == "PCA" or reduction_method == "PCA först, sedan UMAP":
            n_components = st.slider("Välj antal komponenter för PCA", 2, max_components, key="pca_n_components")
            if st.button("Kör PCA"):
                pca = PCA(n_components=n_components)
                reduced_data = pca.fit_transform(data[num_cols])
                st.session_state['pca_df'] = pd.DataFrame(reduced_data, columns=[f"PC{i+1}" for i in range(n_components)])

                st.write(f"PCA-reducerad data med {n_components} komponenter")
                st.dataframe(st.session_state['pca_df'].head())

                explained_variance = pca.explained_variance_ratio_
                st.write("Förklaringsgrad per komponent:")
                st.bar_chart(explained_variance)

                st.subheader("PCA Rapport")
                st.write(f"PCA reducerade datan till {n_components} komponenter med en förklaringsgrad av {explained_variance.sum()*100:.2f}%.")

                if n_components == 2:
                    plt.figure(figsize=(10, 6))
                    sns.scatterplot(x=st.session_state['pca_df']["PC1"], y=st.session_state['pca_df']["PC2"], palette="viridis")
                    plt.title("PCA med 2 komponenter")
                    plt.xlabel(f"PC1 ({explained_variance[0]*100:.2f}% förklarad varians)")
                    plt.ylabel(f"PC2 ({explained_variance[1]*100:.2f}% förklarad varians)")
                    st.pyplot(plt.gcf())

            if reduction_method == "PCA först, sedan UMAP" and 'pca_df' in st.session_state and st.button("Fortsätt med UMAP"):
                data_to_umap = st.session_state['pca_df']

                n_neighbors = st.slider("Välj antal grannar för UMAP", 5, 50, value=15, key="umap_n_neighbors")
                reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors)
                umap_data = reducer.fit_transform(data_to_umap)

                st.write("UMAP-reducerad data efter PCA")
                umap_df = pd.DataFrame(umap_data, columns=["UMAP1", "UMAP2"])
                st.dataframe(umap_df.head())

                plt.figure(figsize=(10, 6))
                sns.scatterplot(x=umap_df["UMAP1"], y=umap_df["UMAP2"], palette="viridis")
                plt.title("UMAP efter PCA, med 2 komponenter")
                plt.xlabel("UMAP1")
                plt.ylabel("UMAP2")
                st.pyplot(plt.gcf())

        elif reduction_method == "UMAP":
            n_neighbors = st.slider("Välj antal grannar för UMAP", 5, 50, value=15, key="umap_n_neighbors")
            if st.button("Kör UMAP"):
                reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors)
                umap_data = reducer.fit_transform(data[num_cols])

                st.write("UMAP-reducerad data")
                umap_df = pd.DataFrame(umap_data, columns=["UMAP1", "UMAP2"])
                st.dataframe(umap_df.head())

                plt.figure(figsize=(10, 6))
                sns.scatterplot(x=umap_df["UMAP1"], y=umap_df["UMAP2"], palette="viridis")
                plt.title("UMAP med 2 komponenter")
                plt.xlabel("UMAP1")
                plt.ylabel("UMAP2")
                st.pyplot(plt.gcf())

    save_dataset(data)
    return data

# ----------------------------------------------------------------------------------------------------------------------------
# Modul 6 - Modellval och Träning
def model_training_module(data):
    st.title("Modellval och Träning")

    # Välj målkategorin (target)
    target_column = st.selectbox("Välj målkategori (target)", data.columns, key="target_column")

    # Välj prediktorer (features)
    features = st.multiselect("Välj features (prediktorer)", data.columns.difference([target_column]), key="feature_columns")

    if len(features) > 0 and target_column:
        X = data[features]
        y = data[target_column]

        # Dela upp data i träning och test
        test_size = st.slider("Välj storlek på testdata (%)", min_value=10, max_value=50, value=20, step=5) / 100
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Modellval
        model_choice = st.selectbox("Välj en modell att träna", ["Logistic Regression", "Random Forest"], key="model_choice")

        if st.button("Träna modell"):
            if model_choice == "Logistic Regression":
                model = LogisticRegression()
            elif model_choice == "Random Forest":
                model = RandomForestClassifier()

            # Träna modellen
            model.fit(X_train, y_train)
            st.success(f"{model_choice} är nu tränad!")

            # Spara den tränade modellen i session_state
            st.session_state["trained_model"] = model

            # Förutsägelser på testdata
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            report = classification_report(y_test, predictions, output_dict=True)

            # Visa noggrannhet och rapport i en DataFrame
            st.write(f"Testdata noggrannhet: {accuracy:.2f}")
            st.subheader("Klassificeringsrapport")
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)

            # Visa confusion matrix
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, predictions)

            # Hantera binär klassificering separat för TP, FP, TN, FN
            if len(cm) == 2:
                tn, fp, fn, tp = cm.ravel()
                cm_df = pd.DataFrame({
                    'True Positives (TP)': [tp],
                    'False Positives (FP)': [fp],
                    'True Negatives (TN)': [tn],
                    'False Negatives (FN)': [fn]
                })
                st.dataframe(cm_df)
            else:
                # För multiklass, visa hela confusion matrix
                plt.figure(figsize=(6, 4))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_)
                plt.xlabel("Prediktioner")
                plt.ylabel("Sanna värden")
                plt.title("Confusion Matrix")
                st.pyplot(plt.gcf())

            # Plot ROC-kurva om Logistic Regression är vald (endast för binär klassificering)
            if model_choice == "Logistic Regression" and len(model.classes_) == 2:
                st.subheader("ROC-kurva")
                y_prob = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)

                plt.figure(figsize=(6, 4))
                plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC-kurva (AUC = {roc_auc:.2f})")
                plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel("Falskt positivt")
                plt.ylabel("Sant positivt")
                plt.title("ROC-kurva")
                plt.legend(loc="lower right")
                st.pyplot(plt.gcf())

            # Ge möjlighet att spara modellen
            model_filename = f"{model_choice}_model.pkl"
            with open(model_filename, "wb") as f:
                joblib.dump(model, f)

            # Ladda ner modellen som en fil
            with open(model_filename, "rb") as f:
                st.download_button(
                    label="Ladda ner tränad modell",
                    data=f,
                    file_name=model_filename,
                    mime="application/octet-stream"
                )

    save_dataset(data)
    return data
# ----------------------------------------------------------------------------------------------------------------------------
# Huvudfunktion som kopplar samman alla moduler med stegvis navigering och bekräftelseknappar
def main():
    st.sidebar.title("Datascience för alla")

    # Kontrollera och initiera tillstånd för varje steg i session_state
    if "data_collected" not in st.session_state:
        st.session_state["data_collected"] = False
    if "eda_done" not in st.session_state:
        st.session_state["eda_done"] = False
    if "preprocessing_done" not in st.session_state:
        st.session_state["preprocessing_done"] = False
    if "feature_engineering_done" not in st.session_state:
        st.session_state["feature_engineering_done"] = False
    if "dimensionality_reduction_done" not in st.session_state:
        st.session_state["dimensionality_reduction_done"] = False
    if "model_trained" not in st.session_state:
        st.session_state["model_trained"] = False

    # Modulval i en specifik ordning
    module = st.sidebar.selectbox("Välj en modul", 
                                  ["Datainsamling", 
                                   "Dataförståelse och EDA", 
                                   "Data Preprocessing", 
                                   "Feature Engineering", 
                                   "Dimensionalitetsreduktion (PCA/UMAP)",  
                                   "Modellval och Träning"], 
                                   key="module_selector")

    # Steg 1: Datainsamling (Första steget som måste genomföras)
    if module == "Datainsamling":
        data = load_data()
        if data is not None:
            st.session_state["data"] = data
            st.session_state["data_collected"] = True
        

    # Steg 2: Dataförståelse och EDA (Tillgänglig efter datainsamling)
    if st.session_state["data_collected"] and module == "Dataförståelse och EDA":
        eda_module(st.session_state["data"])
        st.session_state["eda_done"] = True

    # Kontrollera om datainsamling och EDA är genomförd innan nästa steg
    if st.session_state["eda_done"]:

        # Steg 3: Data Preprocessing (Tillgänglig efter EDA)
        if module == "Data Preprocessing":
            preprocessing_options_selected = preprocessing_module(st.session_state["data"])
            if st.button("Bekräfta Preprocessing"):
                # Nu bearbetas datan baserat på användarens val
                st.session_state["data"] = preprocessing_options_selected
                st.session_state["preprocessing_done"] = True
                st.success("Preprocessing genomförd!")

        # Steg 4: Feature Engineering (Tillgänglig efter preprocessing)
        if st.session_state["preprocessing_done"] and module == "Feature Engineering":
            feature_options_selected = feature_engineering_module(st.session_state["data"])
            if st.button("Bekräfta Feature Engineering"):
                # Nu bearbetas datan baserat på användarens val
                st.session_state["data"] = feature_options_selected
                st.session_state["feature_engineering_done"] = True
                st.success("Feature Engineering genomförd!")

        # Steg 5: Dimensionalitetsreduktion (Tillgänglig efter feature engineering)
        if st.session_state["feature_engineering_done"] and module == "Dimensionalitetsreduktion (PCA/UMAP)":
            dim_reduction_options_selected = dimensionality_reduction_module(st.session_state["data"])
            if st.button("Bekräfta Dimensionalitetsreduktion"):
                # Nu bearbetas datan baserat på användarens val
                st.session_state["data"] = dim_reduction_options_selected
                st.session_state["dimensionality_reduction_done"] = True
                st.success("Dimensionalitetsreduktion genomförd!")

        # Steg 6: Modellval och Träning (Tillgänglig efter dimensionalitetsreduktion)
        if st.session_state["dimensionality_reduction_done"] and module == "Modellval och Träning":
            model_options_selected = model_training_module(st.session_state["data"])
            if st.button("Bekräfta Modellträning"):
                # Nu bearbetas datan baserat på användarens val
                st.session_state["data"] = model_options_selected
                st.session_state["model_trained"] = True
                st.success("Modellträning genomförd!")

        # Steg 7: Modellutvärdering (Tillgänglig efter modellträning)
        if st.session_state["model_trained"] and module == "Modellutvärdering":
            st.session_state["data"] = model_evaluation_module(st.session_state["data"])

if __name__ == "__main__":
    main()