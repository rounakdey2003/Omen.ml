import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.graph_objs as go
import sklearn.neighbors
import streamlit as st
from imblearn.over_sampling import RandomOverSampler
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

with st.spinner('Loading...'):
    st.toast('Checking Environment')


    def main():
        titleCol1, titleCol2, titleCol3 = st.columns([1, 1, 1])
        with titleCol1:
            pass
        with titleCol2:
            st.title('***Lung Cancer Analyser*** 🫁')
            st.write(':grey[Analyse complex data for Lung Cancer disease]')
        with titleCol3:
            pass

        st.divider()

        bodyCol1, bodyCol2 = st.columns([1, 2])

        with bodyCol1:
            with st.container(height=300, border=True):

                if "value" not in st.session_state:
                    st.session_state.value = ":red[Danger Zone]"
                st.header(st.session_state.value)
                st.write(":grey[It may cause data] :red[loss]")
                danger_col1, danger_col2 = st.columns([1, 1])
                with danger_col1:
                    if st.button("Refresh"):
                        st.session_state.value = ":red[Danger Zone]"
                        st.rerun()
                with danger_col2:
                    if st.button(':orange[Clear Cache]'):
                        st.rerun()

                st.warning("***Use only when data overlap occur***")

        with bodyCol2:
            with st.container(height=200, border=True):
                uploaded_file = st.file_uploader(" ", type=["csv"])
                st.markdown(f""":grey[Click on the] **'Browse files'** :grey[button]
                            :grey[ to open file explorer]""")

            if uploaded_file is None:
                st.error("Please import any :red['**.CSV**'] file to start")

            else:
                progress_text = "Reading CSV..."
                my_bar = st.progress(0, text=progress_text)
                for percent_complete in range(100):
                    time.sleep(0.001)
                    my_bar.progress(percent_complete + 1, text=progress_text)
                time.sleep(1)
                my_bar.empty()
                st.success('Data read :green[**Successfully**]')

        with st.expander('Dummy Dataset'):
            with open("csv/lung_cancer_sample.csv", "rb") as file:
                data = file.read()
                st.download_button(
                    label="Download :red[Dataset]",
                    data=data,
                    file_name="lung_cancer_sample.csv",
                    mime="text/csv"
                )

            st.subheader(
                'The dataset contains the following columns, each representing various attributes and conditions related to individuals, along with their lung cancer status.')
            st.write(':orange[***GENDER***]: Gender of the individual (M/F)')
            st.write(':orange[***AGE***]: Age of the individual')
            st.write(':orange[***SMOKING***]: Smoking status (1: No, 2: Yes)')
            st.write(':orange[***YELLOW_FINGERS***]: Presence of yellow fingers (1: No, 2: Yes)')
            st.write(':orange[***ANXIETY***]: Anxiety levels (1: No, 2: Yes)')
            st.write(':orange[***PEER_PRESSURE***]: Influence of peer pressure (1: No, 2: Yes)')
            st.write(':orange[***CHRONIC DISEASE***]: Presence of chronic disease (1: No, 2: Yes)')
            st.write(':orange[***FATIGUE***]: Fatigue levels (1: No, 2: Yes)')
            st.write(':orange[***ALLERGY***]: Presence of allergies (1: No, 2: Yes)')
            st.write(':orange[***WHEEZING***]: Wheezing status (1: No, 2: Yes)')
            st.write(':orange[***ALCOHOL CONSUMING***]: Alcohol consumption status (1: No, 2: Yes)')
            st.write(':orange[***COUGHING***]: Presence of coughing (1: No, 2: Yes)')
            st.write(':orange[***SHORTNESS OF BREATH***]: Shortness of breath status (1: No, 2: Yes)')
            st.write(':orange[***SWALLOWING DIFFICULTY***]: Difficulty in swallowing (1: No, 2: Yes)')
            st.write(':orange[***CHEST PAIN***]: Presence of chest pain (1: No, 2: Yes)')
            st.write(':orange[***LUNG_CANCER***]: Lung cancer diagnosis (YES/NO)')
            st.divider()
            st.subheader('***Procedure***')

            css = """
                                                                       <style>
                                                                       @keyframes float {
                                                                           0% {
                                                                               transform: translateY(0);
                                                                           }
                                                                           50% {
                                                                               transform: translateY(-20px);
                                                                           }
                                                                           100% {
                                                                               transform: translateY(0);
                                                                           }
                                                                       }
    
                                                                       .float-text {
                                                                           animation: float 3s ease-in-out infinite;
                                                                           color: grey;
                                                                       }
                                                                       </style>
                                                                       """
            st.markdown(css, unsafe_allow_html=True)
            text_to_float = "Below processes are only applicable for inbuild dataset. Results may vary in User dataset"
            st.markdown(f'<div class="float-text">{text_to_float}</div>', unsafe_allow_html=True)

            st.write(':blue[***Step 1***]: Download the above Dummy Dataset. [Click on Download Dataset]')
            st.write(
                ':blue[***Step 2***]: Upload the Dummy Dataset. [Click on the Upload button or Drag Drop the downloaded dataset]')
            st.write(
                ':blue[***Step 3***]: You do not have to drop any label, So skip the dropping part. [:red[Disclaimer]: You may have to perform label drop for differnt datasets]')
            st.write(
                ':blue[***Step 4***]: Select the **Lung Cancer** or (Cancer Type) label in Data :grey[Analysis]. [:red[Disclaimer]: Label may vary in different datasets]')
            st.write(
                ':blue[***Step 5***]: Select the **Gender** label in Data :grey[Analysis]. [:red[Disclaimer]: Label may vary in different datasets]')
            st.write(
                ':blue[***Step 6***]: Select the **Age** label in Data :grey[Analysis]. [:red[Disclaimer]: Label may vary in different datasets]')
            st.write(
                ':blue[***Step 7***]: Slide the bar to **0.2 OR 0.3** in Train Test Split area. [:red[Disclaimer]: Test size may vary in different datasets]')
            st.write(
                ':blue[***Step 8***]: Visualize the Interactive Plots, [Info]: Pinch Out to Zoom IN, Pinch In to Zoom Out. OR Select an area to Zoom. Use auto-scale to revert ')

        # Data Editor#

        if uploaded_file:
            st.divider()
            st.title("***Data :grey[Editor]***")
            with st.spinner('Generating...'):

                editorCol1, editorCol2 = st.columns([1, 1])

                with editorCol1:
                    # Data search #
                    with st.expander("Data :green[Search]", expanded=False):
                        st.title("Data :grey[Search]")

                        css = """
                                                    <style>
                                                    @keyframes float {
                                                        0% {
                                                            transform: translateY(0);
                                                        }
                                                        50% {
                                                            transform: translateY(-20px);
                                                        }
                                                        100% {
                                                            transform: translateY(0);
                                                        }
                                                    }
        
                                                    .float-text {
                                                        animation: float 3s ease-in-out infinite;
                                                        color: grey;
                                                    }
                                                    </style>
                                                    """
                        st.markdown(css, unsafe_allow_html=True)
                        text_to_float = "Search data from '.CSV'"
                        st.markdown(f'<div class="float-text">{text_to_float}</div>', unsafe_allow_html=True)

                        container_search = st.container(height=500, border=True)
                        with container_search.container(height=300, border=True):
                            df_original = pd.read_csv(uploaded_file)
                            st.write(df_original)

                        user = container_search.text_input("Enter your search data:")
                        if container_search.button(":green[Search] Data"):
                            if user:
                                st.toast(':orange[Searching Index...]')
                                results = []
                                for index, row in df_original.iterrows():
                                    for column in df_original.columns:
                                        if user.lower() in str(row[column]).lower():
                                            results.append(row)
                                            break
                                if results is not None:
                                    final = pd.DataFrame(results)
                                    st.subheader(":green[Results]")
                                    with st.container(height=300, border=True):

                                        st.write(final)
                                else:
                                    with st.container(height=350, border=True):
                                        st.error("No matching data found")
                            else:
                                with st.container(height=350, border=True):
                                    st.warning("Please enter a search data")

                    # Data Sort #
                    with st.expander("Data :blue[Sort]", expanded=False):
                        st.title("Data :grey[Sort]")

                        css = """
                                                                            <style>
                                                                            @keyframes float {
                                                                                0% {
                                                                                    transform: translateY(0);
                                                                                }
                                                                                50% {
                                                                                    transform: translateY(-20px);
                                                                                }
                                                                                100% {
                                                                                    transform: translateY(0);
                                                                                }
                                                                            }
        
                                                                            .float-text {
                                                                                animation: float 3s ease-in-out infinite;
                                                                                color: grey;
                                                                            }
                                                                            </style>
                                                                            """
                        st.markdown(css, unsafe_allow_html=True)
                        text_to_float = "Sort data from '.CSV'"
                        st.markdown(f'<div class="float-text">{text_to_float}</div>',
                                    unsafe_allow_html=True)

                        container_sort = st.container(height=600, border=True)
                        with container_sort.container(height=300, border=True):
                            st.write(df_original)

                        column = container_sort.selectbox("Select a column to sort by:", df_original.columns)
                        sortCol1, sortCol2 = container_sort.columns([1, 1])

                        with sortCol1:
                            sort_type = st.radio("Select sorting type:", ("Alphabetical", "Numerical"))
                            if sort_type == "Alphabetical":
                                sorted_df = df_original.sort_values(by=column)
                            else:
                                try:
                                    df_original[column] = pd.to_numeric(df_original[column])
                                    sorted_df = df_original.sort_values(by=column)
                                except ValueError:
                                    container_sort.error(
                                        "Selected column contains non-numeric values, unable to sort numerically.")
                                    sorted_df = df_original

                        with sortCol2:
                            sort_order = st.radio("Select sorting order:", ("Ascending", "Descending"))
                            if sort_order == "Descending":
                                sorted_df = sorted_df[::-1]

                        if sort_type or sort_order:
                            progress_text = "Preparing Data..."
                            my_bar = container_sort.progress(0, text=progress_text)
                            for percent_complete in range(100):
                                time.sleep(0.001)
                                my_bar.progress(percent_complete + 1, text=progress_text)
                            time.sleep(1)
                            my_bar.empty()
                            container_sort.download_button("Download :red[CSV]", sorted_df.to_csv(index=False),
                                                           file_name="modified_csv.csv")

                        st.write(":green[Result]")
                        st.write(sorted_df)

                with editorCol2:
                    # Data Edit #
                    with st.expander("Data :red[Edit]", expanded=False):
                        st.title("Data :grey[Edit]")

                        css = """
                                                    <style>
                                                    @keyframes float {
                                                        0% {
                                                            transform: translateY(0);
                                                        }
                                                        50% {
                                                            transform: translateY(-20px);
                                                        }
                                                        100% {
                                                            transform: translateY(0);
                                                        }
                                                    }
        
                                                    .float-text {
                                                        animation: float 3s ease-in-out infinite;
                                                        color: grey;
                                                    }
                                                    </style>
                                                    """
                        st.markdown(css, unsafe_allow_html=True)
                        text_to_float = "Edit data from '.CSV'"
                        st.markdown(f'<div class="float-text">{text_to_float}</div>', unsafe_allow_html=True)

                        container_edit = st.container(height=700, border=True)
                        with container_edit.container(height=300, border=True):
                            st.write(df_original)
                            data_df = df_original.copy()

                        edit_col = container_edit.selectbox("Select column to edit:", options=data_df.columns)
                        edit_index = container_edit.number_input("Enter row index to edit:", min_value=0,
                                                                 max_value=len(data_df) - 1, step=1)
                        edit_value = container_edit.text_input(f"Enter new value:")
                        update = False
                        sidebar_column1, sidebar_column2 = container_edit.columns([2, 1, ])
                        with sidebar_column1:
                            if st.button(":green[Update] Data"):
                                update = True
                                if edit_value:
                                    data_df.at[edit_index, edit_col] = edit_value

                                else:
                                    st.warning("Please enter a value to update.")

                        with sidebar_column2:
                            if update:
                                progress_text = "Preparing Data..."
                                my_bar = container_edit.progress(0, text=progress_text)
                                for percent_complete in range(100):
                                    time.sleep(0.001)
                                    my_bar.progress(percent_complete + 1, text=progress_text)
                                time.sleep(1)
                                my_bar.empty()
                                st.download_button("Download :red[CSV]", data_df.to_csv(index=False),
                                                   file_name="modified_csv.csv")

                        if edit_value:
                            st.write(":green[Result]")

                            st.write(data_df)

                    # Data Delete #
                    with st.expander("Data :red[Delete]", expanded=False):
                        st.title("Data :grey[Delete]")

                        css = """
                                                    <style>
                                                    @keyframes float {
                                                        0% {
                                                            transform: translateY(0);
                                                        }
                                                        50% {
                                                            transform: translateY(-20px);
                                                        }
                                                        100% {
                                                            transform: translateY(0);
                                                        }
                                                    }
        
                                                    .float-text {
                                                        animation: float 3s ease-in-out infinite;
                                                        color: grey;
                                                    }
                                                    </style>
                                                    """
                        st.markdown(css, unsafe_allow_html=True)
                        text_to_float = "Delete data from '.CSV'"
                        st.markdown(f'<div class="float-text">{text_to_float}</div>', unsafe_allow_html=True)

                        container_del = st.container(height=500, border=True)
                        with container_del.container(height=300, border=True):
                            st.write(df_original)
                            delete_df = df_original.copy()
                            selected_indices = container_del.multiselect("Select rows to delete", delete_df.index)
                            delete_df_drop = delete_df.drop(selected_indices)

                            if selected_indices:
                                st.toast(':orange[Removing Index...]')
                                progress_text = "Preparing Data..."
                                my_bar = container_del.progress(0, text=progress_text)
                                for percent_complete in range(100):
                                    time.sleep(0.001)
                                    my_bar.progress(percent_complete + 1, text=progress_text)
                                time.sleep(1)
                                my_bar.empty()
                                container_del.download_button("Download :red[CSV]",
                                                              delete_df_drop.to_csv(index=False),
                                                              file_name="modified_csv.csv", key=0)

                        if selected_indices:
                            st.subheader(":green[Result]")
                            st.write(delete_df_drop)

        # Data Manipulation #

        if uploaded_file:
            st.divider()
            # Overview #
            st.title("***Over:grey[view]***")

            with st.expander("See :red[Documentation]", expanded=False):

                st.subheader(':blue[Describe]')
                st.help(pd.DataFrame.describe)
                st.divider()
                st.subheader(':blue[Drop Na]')
                st.help(pd.DataFrame.dropna)
                st.divider()
                st.subheader(':blue[Drop Duplicates]')
                st.help(pd.DataFrame.drop_duplicates)

            with st.container(height=700, border=True):
                with st.spinner('Generating...'):
                    CleanedNull = df_original.dropna().drop_duplicates()

                    st.toast(':red[Removing Null...]')
                    st.info(":blue[Hint:] Drop unwanted labels.")

                    dropColumn = st.multiselect("**Choose label to drop**", CleanedNull.columns)
                    dropBarData = df_original.drop(columns=dropColumn)

                    CleanedNull = dropBarData

                    # tabular form #
                    tab1, tab2 = st.tabs(["Dataframe", "Describe"])

                    with tab1:
                        st.write(CleanedNull)

                    with tab2:
                        st.write(CleanedNull.describe())

            # -----Data Selection
            st.divider()

            st.title(f"Data :grey[Analysis]")
            with st.expander("See :red[Description]", expanded=False):

                st.write('To effectively analyze the dataset and obtain meaningful results, it is crucial to choose an '
                         'appropriate label. In this dataset, the label we will focus on is "Lung Cancer Diagnosis,'
                         '" which indicates whether a patient has been diagnosed with lung cancer (YES or NO). This label is '
                         'essential as it serves as the target variable for our analysis. By examining the relationship '
                         'between this label and other attributes such as age, gender, smoking habits, and symptoms, '
                         'we can identify patterns and risk factors associated with lung cancer. This focused approach '
                         'enables us to draw actionable insights and develop predictive models to assess cancer risk, '
                         'ultimately helping individuals make informed health decisions.')

            with st.container(height=500, border=True):
                with st.spinner('Generating...'):
                    st.info(':blue[Hint:] Please select the respected label from the dataset.')
                    st.warning(':orange[Warning:] Selecting wrong label may cause major Errors.')

                    cancer = st.selectbox('Select Cancer', CleanedNull.columns)
                    gender = st.selectbox('Select Gender', CleanedNull.columns)
                    age = st.selectbox('Select Age', CleanedNull.columns)

            st.write('##')

            st.header(f" :orange[Encoding of Categorical Columns]")
            with st.expander("See :red[Documentation]", expanded=False):

                st.subheader(':blue[Fit Transform]')
                st.help(LabelEncoder.fit_transform)
            with st.spinner('Generating...'):

                encoder = LabelEncoder()
                CleanedNull[cancer] = encoder.fit_transform(CleanedNull[cancer])
                CleanedNull[gender] = encoder.fit_transform(CleanedNull[gender])
                st.write(CleanedNull)

                con_col = [age]
                cat_col = []
                for i in CleanedNull.columns:
                    if i != age:
                        cat_col.append(i)

            st.write('##')

            st.header(f":orange[Visualization]")

            with st.expander("See :red[Documentation]", expanded=False):

                st.subheader(':blue[Histogram]')
                st.help(px.histogram)
                st.divider()
                st.subheader(':blue[Box]')
                st.help(px.box)
                st.divider()
                st.subheader(':blue[Violin]')
                st.help(px.violin)
                st.divider()
                st.subheader(':blue[Heatmap]')
                st.help(px.imshow)

            st.write('##')

            with st.expander('Visualizing AGE Column'):
                st.subheader(':green[Visualizing AGE Column]')
                with st.spinner('Generating...'):
                    st.write('##')

                    fig1 = px.histogram(CleanedNull, x=age, color=cancer, marginal='box', nbins=50)
                    fig1.update_layout(title=f'Histogram of {age} by {cancer}')

                    fig2 = px.box(CleanedNull, x=cancer, y=age, title=f'Boxplot of {age} by {cancer}')
                    fig2.update_layout(title=f'Boxplot of {age} by {cancer}')

                    st.plotly_chart(fig1, use_container_width=True)
                    st.plotly_chart(fig2, use_container_width=True)

            st.write('##')

            with st.expander('Visualizing Categorical Columns'):
                st.subheader(':green[Visualizing Categorical Columns]')
                with st.spinner('Generating...'):
                    st.write('##')

                    for col in cat_col:
                        st.subheader(f"Distribution of {col}")

                        if cancer in CleanedNull.columns:
                            fig_hue = px.histogram(CleanedNull, x=col, color=cancer)
                            st.plotly_chart(fig_hue)

            st.write('##')

            with st.expander('Visualizing AGE vs Categorical Columns'):
                st.subheader(':green[Visualizing AGE vs Categorical Columns]')
                with st.spinner('Generating...'):
                    st.write('##')

                    figures = []

                    for i in cat_col:
                        fig = go.Figure()

                        fig.add_trace(go.Box(
                            x=CleanedNull[i],
                            y=CleanedNull[age],
                            name=f'AGE vs {i}'
                        ))

                        for value in CleanedNull[cancer].unique():
                            filtered_df = CleanedNull[CleanedNull[cancer] == value]
                            fig.add_trace(go.Box(
                                x=filtered_df[i],
                                y=filtered_df[age],
                                name=f'{age} vs {i} ({cancer}={value})'
                            ))

                        fig.add_trace(go.Violin(
                            x=CleanedNull[i],
                            y=CleanedNull[age],
                            name=f'{age} vs {i} (Violin)',
                            box_visible=True,
                            meanline_visible=True
                        ))

                        fig.update_layout(title=f'Visualizing {age} vs {i}', height=600, width=800)
                        figures.append(fig)

                    for fig in figures:
                        st.plotly_chart(fig)

            st.write('##')

            with st.expander('Visualizing Heatmap'):
                st.subheader(':green[Visualizing Heatmap]')
                with st.spinner('Generating...'):
                    st.write('##')

                    corr_matrix = CleanedNull.corr()

                    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                                    color_continuous_scale="viridis", origin="lower",
                                    labels=dict(x="Features", y="Features", color="Correlation"))

                    fig.update_layout(width=800, height=800)

                    st.plotly_chart(fig)

            st.divider()

            # -----Data TTS
            st.title(f"Data :grey[Preprocessing] ")

            with st.container(height=550, border=True):
                with st.spinner('Generating...'):
                    st.info('Logical replacing column values with (:blue[**0,1**])')

                    X = CleanedNull.drop([cancer], axis=1)
                    y = CleanedNull[cancer]

                    for i in X.columns[2:]:
                        temp = []
                        for j in X[i]:
                            temp.append(j - 1)
                        X[i] = temp
                    st.write(X)

            X_over, y_over = RandomOverSampler().fit_resample(X, y)

            st.write('##')

            st.header(f":orange[Train Test Split]")

            with st.expander("See :red[Documentation]", expanded=False):
                st.subheader(':blue[Train Test Split]')
                st.help(sklearn.model_selection.train_test_split)

            with st.container(height=350):
                with st.spinner('Generating...'):
                    st.info(':blue[Hint:] Select Test size')
                    st.warning(':orange[Warning:] Test size may change expected analysis.')

                    classificationSlider = st.slider("**Test size**", 0.1, 0.5)
                    st.toast(':orange[Test Size Modified...]')

                    X_train, X_test, y_train, y_test = train_test_split(X_over, y_over, test_size=classificationSlider,
                                                                        random_state=42, stratify=y_over)
                    st.write(f'Train shape : {X_train.shape}')
                    st.write(f'Test shape: {X_test.shape}')

            st.write('##')

            st.header(f":orange[Scaling]")

            with st.expander("See :red[Documentation]", expanded=False):

                st.subheader(':blue[Standard Scaler]')
                st.help(sklearn.preprocessing.StandardScaler)
                st.divider()
                st.subheader(':blue[Transform]')
                st.help(StandardScaler.transform)
                st.divider()
                st.subheader(':blue[Fit Transform]')
                st.help(StandardScaler.fit_transform)

            with st.container(height=500, border=True):
                with st.spinner('Generating...'):
                    scaler = StandardScaler()
                    X_train[age] = scaler.fit_transform(X_train[[age]])
                    X_test[age] = scaler.transform(X_test[[age]])
                    st.write(X_train)

            st.write('##')

            st.header(f":orange[Model Building]")

            with st.expander("See :red[Documentation]", expanded=False):

                st.subheader(':blue[K-Nearest Neighbors Classifier]')
                st.help(sklearn.neighbors.KNeighborsClassifier)
                st.divider()
                st.subheader(':blue[Support Vector Machine]')
                st.help(sklearn.model_selection.RandomizedSearchCV)
                st.divider()
                st.subheader(':blue[Logistic Regression]')
                st.help(sklearn.linear_model.LogisticRegression)
                st.divider()
                st.subheader(':blue[Random Forest Classifier]')
                st.help(sklearn.ensemble.RandomForestClassifier)
                st.divider()
                st.subheader(':blue[Gradient Boosting Classifier]')
                st.help(sklearn.ensemble.GradientBoostingClassifier)
                st.divider()
                st.subheader(':blue[LGBM Classifier]')
                st.help(lgb.LGBMClassifier)
                st.divider()
                st.subheader(':blue[Selected Model - SVC]')
                st.help(sklearn.svm.SVC)
                st.divider()
                st.subheader(':blue[AUC & ROC Curve]')
                st.help(metrics.roc_curve)

            st.write('##')

            with st.expander('K-Nearest Neighbors Classifier'):
                st.subheader(":green[K-Nearest Neighbors Classifier]")
                with st.spinner('Generating...'):
                    st.write('##')

                    knn_scores = []
                    for k in range(1, 21):
                        knn = KNeighborsClassifier(n_neighbors=k)
                        scores = cross_val_score(knn, X_train, y_train, cv=5)
                        knn_scores.append(scores.mean())

                    x_ticks = list(range(1, 21))
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=x_ticks, y=knn_scores, mode='lines+markers', name='KNN Score'))

                    fig.update_layout(
                        title="KNN Cross-Validation Scores for Different k",
                        xaxis=dict(title='Number of Neighbors (k)', tickmode='linear'),
                        yaxis=dict(title='Cross-Validation Score'),
                        xaxis_tickvals=x_ticks
                    )

                    st.plotly_chart(fig)

                    knn = KNeighborsClassifier(n_neighbors=1)
                    knn.fit(X_train, y_train)

                    y_pred = knn.predict(X_test)

                    confusion_knn = confusion_matrix(y_test, y_pred)

                    fig = ff.create_annotated_heatmap(
                        z=confusion_knn,
                        x=['Predicted Class 1', 'Predicted Class 2'],
                        y=['Actual Class 1', 'Actual Class 2'],
                        annotation_text=confusion_knn,
                        colorscale='Viridis'
                    )

                    fig.update_layout(title_text='Confusion Matrix', xaxis_title='Predicted', yaxis_title='Actual')

                    st.plotly_chart(fig)

                    st.write("Classification Report:")
                    st.text(classification_report(y_test, y_pred))

            st.write('##')

            with st.expander('Support Vector Machine'):
                st.subheader(":green[Support Vector Machine]")
                with st.spinner('Generating...'):
                    st.write('##')

                    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
                    rcv = RandomizedSearchCV(SVC(), param_grid, cv=5)
                    rcv.fit(X_train, y_train)
                    y_pred_svc = rcv.predict(X_test)

                    confusion_svc = confusion_matrix(y_test, y_pred_svc)

                    z = confusion_svc.tolist()
                    x = ['Predicted ' + str(i) for i in range(len(z))]
                    y = ['Actual ' + str(i) for i in range(len(z))]

                    fig = ff.create_annotated_heatmap(z, x=x, y=y, colorscale='Viridis')
                    fig.update_layout(title='Confusion Matrix', xaxis_title='Predicted', yaxis_title='Actual')

                    st.plotly_chart(fig)
                    st.write('Classification Report:')
                    st.text(classification_report(y_test, y_pred_svc))
                    st.write(f'Best Parameters of SVC model: {rcv.best_params_}')

            st.write('##')

            with st.expander('Logistic Regression'):
                st.subheader(":green[Logistic Regression]")
                with st.spinner('Generating...'):
                    st.write('##')

                    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
                                  'max_iter': [50, 75, 100, 200, 300, 400, 500, 700]}
                    log = RandomizedSearchCV(LogisticRegression(solver='lbfgs'), param_grid, cv=5)
                    log.fit(X_train, y_train)
                    y_pred_log = log.predict(X_test)

                    confusion_log = confusion_matrix(y_test, y_pred_log)
                    report = classification_report(y_test, y_pred_log, output_dict=True)

                    st.write("Confusion Matrix:")
                    st.write(confusion_log)

                    fig = go.Figure(data=go.Heatmap(
                        z=confusion_log,
                        x=['Predicted 0', 'Predicted 1'],
                        y=['Actual 0', 'Actual 1'],
                        hoverongaps=False,
                        colorscale='Viridis'))

                    fig.update_layout(title='Confusion Matrix Heatmap',
                                      xaxis=dict(title='Predicted Labels'),
                                      yaxis=dict(title='Actual Labels'))

                    st.plotly_chart(fig)

                    st.write("Classification Report:")
                    report_df = pd.DataFrame(report).transpose()
                    st.write(report_df)

            st.write('##')

            with st.expander('Random Forest Classifier'):
                st.subheader(":green[Random Forest Classifier]")
                with st.spinner('Generating...'):
                    st.write('##')

                    param_grid = {
                        'n_estimators': [50, 75, 100, 150, 200, 300],
                    }

                    rcv = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
                    rcv.fit(X_train, y_train)

                    y_pred_rcv = rcv.predict(X_test)

                    confusion_rcv = confusion_matrix(y_test, y_pred_rcv)

                    z = confusion_rcv
                    x = ['Predicted ' + str(label) for label in range(confusion_rcv.shape[1])]
                    y = ['Actual ' + str(label) for label in range(confusion_rcv.shape[0])]

                    fig = ff.create_annotated_heatmap(z, x=x, y=y, colorscale='Viridis', showscale=True)

                    st.write("Classification Report")
                    st.text(classification_report(y_test, y_pred_rcv))

                    st.write("Confusion Matrix")
                    st.plotly_chart(fig)

                    st.write("Best Parameter")
                    st.write(f'Best Parameter: {rcv.best_params_}')

            st.write('##')

            with st.expander('Gradient Boosting Classifier'):
                st.subheader(":green[Gradient Boosting Classifier]")
                with st.spinner('Generating...'):
                    st.write('##')

                    param_grid = {
                        'learning_rate': [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1],
                        'n_estimators': [50, 75, 100, 150, 200, 300],
                    }

                    gbc = RandomizedSearchCV(GradientBoostingClassifier(random_state=42), param_grid, cv=5)
                    gbc.fit(X_train, y_train)
                    y_pred_gbc = gbc.predict(X_test)

                    confusion_gbc = confusion_matrix(y_test, y_pred_gbc)

                    num_classes = len(set(y))
                    labels = [f'Class {i}' for i in range(num_classes)]

                    z = confusion_gbc
                    x = [f'Predicted {label}' for label in labels]
                    y = [f'Actual {label}' for label in labels]

                    fig = ff.create_annotated_heatmap(z, x=x, y=y, colorscale='Viridis')

                    st.write("Confusion Matrix")
                    st.plotly_chart(fig)

                    st.write("Classification Report")
                    report = classification_report(y_test, y_pred_gbc, output_dict=True)
                    df_report = pd.DataFrame(report).transpose()
                    st.dataframe(df_report)

                    st.write("Best Parameters")
                    st.write(gbc.best_params_)

            st.write('##')

            with st.expander('LGBM Classifier'):
                st.subheader(":green[LGBM Classifier]")
                with st.spinner('Generating...'):
                    st.write('##')

                    model = lgb.LGBMClassifier()
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    confusion = confusion_matrix(y_test, y_pred)

                    fig = go.Figure(data=go.Heatmap(
                        z=confusion,
                        x=np.unique(y_test),
                        y=np.unique(y_test),
                        colorscale='Blues',
                        showscale=True,
                        hoverongaps=False
                    ))

                    fig.update_layout(
                        title='Confusion Matrix',
                        xaxis_title='Predicted',
                        yaxis_title='Actual',
                        autosize=True
                    )

                    st.plotly_chart(fig)

                    report = classification_report(y_test, y_pred, output_dict=True)
                    st.write(pd.DataFrame(report).transpose())

            st.write('##')

            with st.expander('Selected Model - SVC'):
                st.subheader(":green[Selected Model - SVC]")
                with st.spinner('Generating...'):
                    st.write('##')

                    model = SVC(gamma=10, C=100)
                    model.fit(X_train, y_train)
                    y_pred_svc = model.predict(X_test)

                    confusion_svc = confusion_matrix(y_test, y_pred_svc)

                    z = confusion_svc
                    x = ['Predicted 0', 'Predicted 1']
                    y = ['Actual 0', 'Actual 1']
                    z_text = [[str(y) for y in x] for x in z]

                    fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text, colorscale='Viridis')

                    st.plotly_chart(fig)
                    st.text('Confusion Matrix:')
                    st.dataframe(pd.DataFrame(confusion_svc, index=y, columns=x))
                    st.text('Classification Report:')
                    st.text(classification_report(y_test, y_pred_svc))

            st.write('##')

            with st.expander('AUC & ROC Curve'):
                st.subheader(":green[AUC & ROC Curve]")
                with st.spinner('Generating...'):
                    st.write('##')

                    auc = metrics.roc_auc_score(y_test, y_pred_svc)
                    false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y_test, y_pred_svc)

                    fig = go.Figure()

                    fig.add_trace(go.Scatter(
                        x=false_positive_rate,
                        y=true_positive_rate,
                        mode='lines',
                        name='ROC Curve',
                        line=dict(color='blue', dash='dash')
                    ))

                    fig.add_trace(go.Scatter(
                        x=false_positive_rate,
                        y=true_positive_rate,
                        fill='tozeroy',
                        fillcolor='lightblue',
                        mode='none',
                        showlegend=False
                    ))

                    fig.update_layout(
                        title=f"AUC & ROC Curve (AUC = {auc:.4f})",
                        xaxis=dict(title='False Positive Rate', range=[0, 1]),
                        yaxis=dict(title='True Positive Rate', range=[0, 1]),
                        showlegend=False
                    )

                    st.plotly_chart(fig)


    if __name__ == "__main__":
        main()

    st.toast(':green[Ready!]')
