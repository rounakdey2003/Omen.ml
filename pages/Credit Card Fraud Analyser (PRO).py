import time
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.graph_objs as go
import sklearn.model_selection
import streamlit as st
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
from imblearn.under_sampling import NearMiss
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

with st.spinner('Loading...'):
    st.toast('Checking Environment')


    def main():
        titleCol1, titleCol2, titleCol3 = st.columns([1, 1, 1])
        with titleCol1:
            pass
        with titleCol2:
            st.title('***Credit Card Fraud Analyser*** 💳')
            st.write(':grey[Analyse complex data for credit card fraud]')
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
            with open("csv/credit_card_sample.csv", "rb") as file:
                data = file.read()
                st.download_button(
                    label="Download :red[Dataset]",
                    data=data,
                    file_name="credit_card_sample.csv",
                    mime="text/csv"
                )

            st.subheader('The dataset appears to be a sample of credit card transactions with the following columns.')
            st.write(':orange[***Time***]: The time elapsed since the first transaction in the dataset.')
            st.write(
                ':orange[***V2, V4, V10, V11, V12, V14, V17, V19***]: These are anonymized feature variables resulting from a PCA transformation.')
            st.write(':orange[***Amount***]: The transaction amount.')
            st.write(
                ':orange[***Class***]: The target variable indicating if the transaction is fraudulent (1) or not (0).')
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
                ':blue[***Step 4***]: Select the **Class** label in Visualization Setup :grey[(Class)]. (Info: Label which contains the fraud and non fraud data)')
            st.write(
                ':blue[***Step 5***]: Select the **Time** label in Visualization Setup :grey[(Time/Amount)]. [:red[Disclaimer]: Label may vary in different datasets]')
            st.write(
                ':blue[***Step 6***]: Select the **Amoun**t label in Visualization Setup :grey[(Time/Amount)]. [:red[Disclaimer]: Label may vary in different datasets]')
            st.write(
                ':blue[***Step 7***]: Select **V17**, **V14**, **V12**, **V10** as Negative Correlation label in Negative Correlations :grey[Setup]. [:red[Disclaimer]: Label may vary in different datasets]')
            st.write(
                ':blue[***Step 8***]: Select **V11**, **V4**, **V2**, **V19** as Positive Correlation label in Positive Correlations :grey[Setup]. [:red[Disclaimer]: Label may vary in different datasets]')
            st.write(
                ':blue[***Step 9***]: Select **V14**, **V12**, **V10** as Fraud Detection label in Fraud Detection :grey[Setup]. [Info]: Label is related to the Negative Correlation')
            st.write(
                ':blue[***Step 10***]: Slide the bar to **0.2** in Classifier :grey[Setup]. [:red[Disclaimer]: Test size may vary in different datasets]')
            st.write(
                ':blue[***Step 11***]: Visualize the Interactive Plots, [Info]: Pinch Out to Zoom IN, Pinch In to Zoom Out. OR Select an area to Zoom. Use auto-scale to revert ')

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

            with st.container(height=800, border=True):
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
                        st.subheader("Dataframe")
                        st.write(CleanedNull)

                    with tab2:
                        st.subheader("Description")
                        st.write(CleanedNull.describe())

            st.divider()

            # -----Data Selection

            st.header(f"Visualization Setup :grey[(Class)]")
            with st.expander("See :red[Documentation]", expanded=False):

                st.subheader(':blue[Histogram]')
                st.help(px.histogram)

            with st.container(height=300, border=True):
                with st.spinner('Generating...'):
                    st.info(':blue[Hint:] Please select the respected label from the dataset.')
                    st.warning(':orange[Warning:] Selecting wrong label may cause major Errors.')

                    fraud_class = st.selectbox('Select Fraud Class', CleanedNull.columns)

            st.write('##')

            with st.expander('Visualizing Distributions'):
                st.subheader(f':green[Visualizing Distributions]')
                with st.spinner('Generating...'):
                    st.write('##')

                    colors = ['#1f77b4', '#ff7f0e']

                    fig = px.histogram(CleanedNull, x=fraud_class, color=fraud_class, color_discrete_sequence=colors,
                                       title=f'Distributions of {fraud_class}',
                                       labels={'Class': fraud_class, 'count': 'Count'})

                    st.plotly_chart(fig)

            st.divider()

            st.header(f"Visualization Setup :grey[(Time/Amount)]")
            with st.container(height=400, border=True):
                with st.spinner('Generating...'):
                    st.info(':blue[Hint:] Please select the respected label from the dataset.')
                    st.warning(':orange[Warning:] Selecting wrong label may cause major Errors.')

                    Time = st.selectbox('Select Time', CleanedNull.columns)
                    Amount = st.selectbox('Select Amount', CleanedNull.columns)
                    amount = CleanedNull[Amount].astype(int)
                    timee = CleanedNull[Time].astype(int)

            st.write('##')

            with st.expander('Visualizing Distributions'):
                st.subheader(f':green[Visualizing Distributions]')
                with st.spinner('Generating...'):
                    st.write('##')

                    fig = go.Figure()

                    fig.add_trace(go.Histogram(
                        x=amount,
                        marker_color='cyan',
                        name=f'Transaction {Amount}',
                        nbinsx=50
                    ))

                    fig.update_layout(
                        title_text=f'Distribution of {Amount}',
                        xaxis_title_text=f'{Amount}',
                        yaxis_title_text='Count',
                        bargap=0.2,
                        bargroupgap=0.1
                    )

                    fig.add_trace(go.Histogram(
                        x=timee,
                        marker_color='pink',
                        name=f'Transaction {Time}',
                        nbinsx=50
                    ))

                    fig.update_layout(
                        title_text=f'Distribution of {Amount} and {Time}',
                        xaxis_title_text='Value',
                        yaxis_title_text='Count',
                        bargap=0.2,
                        bargroupgap=0.1
                    )

                    st.plotly_chart(fig)

            st.divider()

            st.header(f" :orange[Scaling Categorical Columns]")
            with st.expander("See :red[Documentation]", expanded=False):

                st.subheader(':blue[Fit Transform]')
                st.help(RobustScaler.fit_transform)
            with st.spinner('Generating...'):
                std_scaler = StandardScaler()
                rob_scaler = RobustScaler()

                CleanedNull['scaled_amount'] = rob_scaler.fit_transform(CleanedNull[Amount].values.reshape(-1, 1))
                CleanedNull['scaled_time'] = rob_scaler.fit_transform(CleanedNull[Time].values.reshape(-1, 1))

                CleanedNull.drop([Time, Amount], axis=1, inplace=True)

                scaled_amount = CleanedNull['scaled_amount']
                scaled_time = CleanedNull['scaled_time']

                CleanedNull.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
                CleanedNull.insert(0, 'scaled_amount', scaled_amount)
                CleanedNull.insert(1, 'scaled_time', scaled_time)
                st.write(CleanedNull)

            st.divider()

            st.header(f" :orange[Data Splitting]")
            with st.expander("See :red[Documentation]", expanded=False):
                st.subheader(':blue[Train Test Split]')
                st.help(sklearn.model_selection.train_test_split)
                st.divider()
                st.subheader(':blue[Stratified Shuffle Split]')
                st.help(sklearn.model_selection.StratifiedShuffleSplit)

            with st.container(height=300, border=True):
                with st.spinner('Generating...'):
                    st.write('No Frauds: ',
                             round(CleanedNull[fraud_class].value_counts()[0] / len(CleanedNull) * 100, 2),
                             '% of the dataset')
                    st.write('Frauds: ', round(CleanedNull[fraud_class].value_counts()[1] / len(CleanedNull) * 100, 2),
                             '% of the dataset')

                    X = CleanedNull.drop(fraud_class, axis=1)
                    y = CleanedNull[fraud_class]

                    sss = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

                    for train_index, test_index in sss.split(X, y):
                        original_Xtrain, original_Xtest = X.iloc[train_index], X.iloc[test_index]
                        original_ytrain, original_ytest = y.iloc[train_index], y.iloc[test_index]

                    original_Xtrain = original_Xtrain.values
                    original_Xtest = original_Xtest.values
                    original_ytrain = original_ytrain.values
                    original_ytest = original_ytest.values

                    train_unique_label, train_counts_label = np.unique(original_ytrain, return_counts=True)
                    test_unique_label, test_counts_label = np.unique(original_ytest, return_counts=True)

                    st.write(':green[***Label Distributions***]')
                    label_dist_Col1, label_dist_Col2 = st.columns([1, 1])
                    with label_dist_Col1:
                        st.write(train_counts_label / len(original_ytrain))
                    with label_dist_Col2:
                        st.write(test_counts_label / len(original_ytest))

            st.write('##')

            st.header(f" :orange[Random Under Sampling]")
            with st.expander("See :red[Description]", expanded=False):
                st.write(
                    'Random Under-Sampling involves reducing the amount of data to achieve a more balanced dataset, which helps prevent model overfitting.'
                    ' The main drawback of Random Under-Sampling is the potential loss of valuable information. ***--> Identify the number of fraud transactions Fraud = 1.'
                    '--> Match the number of non-fraud transactions to the number of fraud transactions for a 50/50 ratio.'
                    '--> Implement the random under-sampling to create a sub-sample . --> Shuffle the balanced dataset to ensure that models can maintain accuracy consistently across runs.***')
            with st.spinner('Generating...'):
                CleanedNull = CleanedNull.sample(frac=1)

                fraud_df = CleanedNull.loc[CleanedNull[fraud_class] == 1]
                non_fraud_df = CleanedNull.loc[CleanedNull[fraud_class] == 0][:492]

                normal_distributed_df = pd.concat([fraud_df, non_fraud_df])

                new_CleanedNull = normal_distributed_df.sample(frac=1, random_state=42)
                st.write(new_CleanedNull)

            ######
            st.write('##')

            with st.expander('Visualizing Equal Distribution'):
                st.subheader(f':green[Visualizing Equal Distribution]')
                with st.spinner('Generating...'):
                    st.write('##')

                    class_distribution = new_CleanedNull[fraud_class].value_counts(normalize=True)
                    st.write(class_distribution)

                    fig = px.bar(class_distribution,
                                 x=class_distribution.index,
                                 y=class_distribution.values,
                                 labels={'x': 'Class', 'y': 'Proportion'},
                                 title='Equally Distributed Classes')

                    st.plotly_chart(fig)

            st.write('##')

            with st.expander('Visualizing Correlation Matrix'):
                st.subheader(f':green[Visualizing Correlation Matrix]')
                with st.spinner('Generating...'):
                    st.write('##')

                    colors = ['#1f77b4', '#ff7f0e']

                    corr = CleanedNull.corr()
                    sub_sample_corr = new_CleanedNull.corr()

                    fig1 = px.imshow(corr, color_continuous_scale=colors, aspect='auto')
                    fig1.update_layout(title="Imbalanced Correlation Matrix (don't use for reference)", width=800,
                                       height=600)

                    fig2 = px.imshow(sub_sample_corr, color_continuous_scale=colors, aspect='auto')
                    fig2.update_layout(title='SubSample Correlation Matrix (use for reference)', width=800, height=600)

                    st.plotly_chart(fig1)
                    st.plotly_chart(fig2)

            st.divider()

            st.header('Negative Correlations :grey[Setup]')
            with st.container(height=500, border=True):
                with st.spinner('Generating...'):
                    st.info(':blue[Hint:] Please select the required label from the dataset.')
                    st.warning(':orange[Warning:] Selecting wrong label may cause analysis errors.')

                    plotly_colors = px.colors.qualitative.Plotly

                    v1 = st.selectbox('Select 1st Variation', CleanedNull.columns, key=1)
                    v2 = st.selectbox('Select 2nd Variation', CleanedNull.columns, key=2)
                    v3 = st.selectbox('Select 3rd Variation', CleanedNull.columns, key=3)
                    v4 = st.selectbox('Select 4th Variation', CleanedNull.columns, key=4)

            st.write('##')
            with st.expander('Visualizing Negative Correlations'):
                st.subheader(f':green[Visualizing Negative Correlations]')
                with st.spinner('Generating...'):
                    st.write('##')

                    titles = [f'{v1} vs Class Negative Correlation',
                              f'{v2} vs Class Negative Correlation',
                              f'{v3} vs Class Negative Correlation',
                              f'{v4} vs Class Negative Correlation']

                    columns = [v1, v2, v3, v4]

                    for col, title in zip(columns, titles):
                        fig = px.box(new_CleanedNull, x=fraud_class, y=col, color=fraud_class, title=title,
                                     color_discrete_sequence=plotly_colors)
                        st.plotly_chart(fig)

            st.write('##')

            st.header('Positive Correlations :grey[Setup]')
            with st.container(height=500, border=True):
                with st.spinner('Generating...'):
                    st.info(':blue[Hint:] Please select the required label from the dataset.')
                    st.warning(':orange[Warning:] Selecting wrong label may cause analysis errors.')

                    v5 = st.selectbox('Select 1st Variation', CleanedNull.columns, key=5)
                    v6 = st.selectbox('Select 2nd Variation', CleanedNull.columns, key=6)
                    v7 = st.selectbox('Select 3rd Variation', CleanedNull.columns, key=7)
                    v8 = st.selectbox('Select 4th Variation', CleanedNull.columns, key=8)

            st.write('##')

            with st.expander('Visualizing Positive Correlations'):
                st.subheader(f':green[Visualizing Positive Correlations]')
                with st.spinner('Generating...'):
                    st.write('##')

                    figs = []

                    features = [v5, v6, v7, v8]

                    for feature in features:
                        fig = px.box(new_CleanedNull, x=fraud_class, y=feature, color=fraud_class,
                                     title=f'{feature} vs Class Positive Correlation')
                        figs.append(fig)

                    for fig in figs:
                        st.plotly_chart(fig)

            st.divider()

            st.header('Fraud Detection :grey[Setup]')
            with st.container(height=500, border=True):
                with st.spinner('Generating...'):
                    st.info(':blue[Hint:] Please select the required label from the dataset.')
                    st.warning(':orange[Warning:] Selecting wrong label may cause analysis errors.')

                    v_dist1 = st.selectbox('Select 1st Variation', CleanedNull.columns, key=9)
                    v_dist2 = st.selectbox('Select 2nd Variation', CleanedNull.columns, key=10)
                    v_dist3 = st.selectbox('Select 3rd Variation', CleanedNull.columns, key=11)

            st.write('##')

            with st.expander('Visualizing Fraud Detection'):
                st.subheader(f':green[Visualizing Fraud Detection]')
                with st.spinner('Generating...'):
                    st.write('##')

                    fraud_df = new_CleanedNull[new_CleanedNull[fraud_class] == 1]

                    v_fig1 = px.histogram(fraud_df, x=v_dist1, nbins=50,
                                          title=f'{v_dist1} Distribution (Fraud Transactions)')
                    st.plotly_chart(v_fig1)

                    v_fig2 = px.histogram(fraud_df, x=v_dist2, nbins=50,
                                          title=f'{v_dist2} Distribution (Fraud Transactions)')
                    st.plotly_chart(v_fig2)

                    v_fig3 = px.histogram(fraud_df, x=v_dist3, nbins=50,
                                          title=f'{v_dist3} Distribution (Fraud Transactions)')
                    st.plotly_chart(v_fig3)

            ##########
            st.write('##')

            with st.expander('Fraud Information'):
                st.subheader(f':green[Fraud Information]')
                with st.spinner('Generating...'):
                    st.write('##')
                    v_fraud1 = new_CleanedNull[v_dist1].loc[new_CleanedNull[fraud_class] == 1].values
                    q25, q75 = np.percentile(v_fraud1, 25), np.percentile(v_fraud1, 75)
                    st.write('Quartile 25: {} | Quartile 75: {}'.format(q25, q75))
                    v_fraud1_iqr = q75 - q25
                    st.write('iqr: {}'.format(v_fraud1_iqr))

                    v_fraud1_cut_off = v_fraud1_iqr * 1.5
                    v_fraud1_lower, v_fraud1_upper = q25 - v_fraud1_cut_off, q75 + v_fraud1_cut_off
                    st.write('Cut Off: {}'.format(v_fraud1_cut_off))
                    st.write('{} Lower: {}'.format(v_dist1, v_fraud1_lower))
                    st.write('{} Upper: {}'.format(v_dist1, v_fraud1_upper))

                    outliers = [x for x in v_fraud1 if x < v_fraud1_lower or x > v_fraud1_upper]
                    st.write('Feature {} Outliers for Fraud Cases: {}'.format(v_dist1, len(outliers)))
                    st.write('{} outliers:{}'.format(v_dist1, outliers))

                    new_CleanedNull = new_CleanedNull.drop(
                        new_CleanedNull[(new_CleanedNull[v_dist1] > v_fraud1_upper) | (
                                new_CleanedNull[v_dist1] < v_fraud1_lower)].index)
                    st.write('##')

                    v_fraud2 = new_CleanedNull[v_dist2].loc[new_CleanedNull[fraud_class] == 1].values
                    q25, q75 = np.percentile(v_fraud2, 25), np.percentile(v_fraud2, 75)
                    v_fraud2_iqr = q75 - q25

                    v_fraud2_cut_off = v_fraud2_iqr * 1.5
                    v_fraud2_lower, v_fraud2_upper = q25 - v_fraud2_cut_off, q75 + v_fraud2_cut_off
                    st.write('{} Lower: {}'.format(v_dist2, v_fraud2_lower))
                    st.write('{} Upper: {}'.format(v_dist2, v_fraud2_upper))
                    outliers = [x for x in v_fraud2 if x < v_fraud2_lower or x > v_fraud2_upper]
                    st.write('{} outliers: {}'.format(v_dist2, outliers))
                    st.write('Feature {} Outliers for Fraud Cases: {}'.format(v_dist2, len(outliers)))
                    new_CleanedNull = new_CleanedNull.drop(
                        new_CleanedNull[(new_CleanedNull[v_dist2] > v_fraud2_upper) | (
                                new_CleanedNull[v_dist2] < v_fraud2_lower)].index)
                    st.write('Number of Instances after outliers removal: {}'.format(len(new_CleanedNull)))
                    st.write('##')

                    v_fraud3 = new_CleanedNull[v_dist3].loc[new_CleanedNull[fraud_class] == 1].values
                    q25, q75 = np.percentile(v_fraud3, 25), np.percentile(v_fraud3, 75)
                    v_fraud3_iqr = q75 - q25

                    v_fraud3_cut_off = v_fraud3_iqr * 1.5
                    v_fraud3_lower, v_fraud3_upper = q25 - v_fraud3_cut_off, q75 + v_fraud3_cut_off
                    st.write('{} Lower: {}'.format(v_dist3, v_fraud3_lower))
                    st.write('{} Upper: {}'.format(v_dist3, v_fraud3_upper))
                    outliers = [x for x in v_fraud3 if x < v_fraud3_lower or x > v_fraud3_upper]
                    st.write('{} outliers: {}'.format(v_dist3, outliers))
                    st.write('Feature {} Outliers for Fraud Cases: {}'.format(v_dist3, len(outliers)))
                    new_CleanedNull = new_CleanedNull.drop(
                        new_CleanedNull[(new_CleanedNull[v_dist3] > v_fraud3_upper) | (
                                new_CleanedNull[v_dist3] < v_fraud3_lower)].index)
                    st.write('Number of Instances after outliers removal: {}'.format(len(new_CleanedNull)))

            st.write('##')

            with st.expander('Visualizing Outlines Reduction'):
                st.subheader(f':green[Visualizing Outlines Reduction]')
                with st.spinner('Generating...'):
                    st.write('##')

                    colors = ['#B3F9C5', '#f9c5b3']

                    fig1 = px.box(new_CleanedNull, x=fraud_class, y=v_dist1, color=fraud_class,
                                  color_discrete_sequence=colors,
                                  title=f"{v_dist1} Feature - Reduction of outliers")
                    fig1.update_layout(annotations=[dict(text='Fewer extreme outliers', x=0.98, y=-17.5,
                                                         showarrow=True, arrowhead=1, ax=0, ay=50)])

                    fig2 = px.box(new_CleanedNull, x=fraud_class, y=v_dist2, color=fraud_class,
                                  color_discrete_sequence=colors,
                                  title=f"{v_dist2} Feature - Reduction of outliers")
                    fig2.update_layout(annotations=[dict(text='Fewer extreme outliers', x=0.98, y=-17.3,
                                                         showarrow=True, arrowhead=1, ax=0, ay=50)])

                    fig3 = px.box(new_CleanedNull, x=fraud_class, y=v_dist3, color=fraud_class,
                                  color_discrete_sequence=colors,
                                  title=f"{v_dist3} Feature - Reduction of outliers")
                    fig3.update_layout(annotations=[dict(text='Fewer extreme outliers', x=0.95, y=-16.5,
                                                         showarrow=True, arrowhead=1, ax=0, ay=50)])

                    st.plotly_chart(fig1)
                    st.plotly_chart(fig2)
                    st.plotly_chart(fig3)

            st.divider()

            st.header(f" :orange[Dimensionality Reduction and Clustering]")
            with st.expander("See :red[Documentation]", expanded=False):
                st.subheader(':blue[TSNE]')
                st.help(sklearn.manifold._t_sne)
                st.divider()
                st.subheader(':blue[PCA]')
                st.help(sklearn.decomposition._pca)
                st.divider()
                st.subheader(':blue[TruncatedSVD]')
                st.help(sklearn.decomposition._truncated_svd)

            with st.container(height=150):
                with st.spinner('Generating...'):
                    X = new_CleanedNull.drop(fraud_class, axis=1)
                    y = new_CleanedNull[fraud_class]

                    t0 = time.time()
                    X_reduced_tsne = TSNE(n_components=2, random_state=42).fit_transform(X.values)
                    t1 = time.time()
                    st.write("T-SNE took {:.2} s".format(t1 - t0))

                    t0 = time.time()
                    X_reduced_pca = PCA(n_components=2, random_state=42).fit_transform(X.values)
                    t1 = time.time()
                    st.write("PCA took {:.2} s".format(t1 - t0))

                    t0 = time.time()
                    X_reduced_svd = TruncatedSVD(n_components=2, algorithm='randomized', random_state=42).fit_transform(
                        X.values)
                    t1 = time.time()
                    st.write("Truncated SVD took {:.2} s".format(t1 - t0))

            st.write('##')

            with st.expander('Visualizing Clustering using Dimensionality Reduction'):
                st.subheader(f':green[Visualizing Clustering using Dimensionality Reduction]')
                with st.spinner('Generating...'):
                    st.write('##')

                    X_reduced_tsne = np.random.randn(100, 2)
                    X_reduced_pca = np.random.randn(100, 2)
                    X_reduced_svd = np.random.randn(100, 2)
                    y = np.random.randint(0, 2, 100)

                    color_map = {0: '#0A0AFF', 1: '#AF0000'}

                    fig_tsne = px.scatter(
                        x=X_reduced_tsne[:, 0], y=X_reduced_tsne[:, 1],
                        color=[color_map[label] for label in y],
                        title='t-SNE',
                        labels={'x': 'Component 1', 'y': 'Component 2'}
                    )
                    fig_tsne.update_traces(marker=dict(size=5, line=dict(width=2, color='DarkSlateGrey')))
                    fig_tsne.update_layout(showlegend=False)

                    fig_pca = px.scatter(
                        x=X_reduced_pca[:, 0], y=X_reduced_pca[:, 1],
                        color=[color_map[label] for label in y],
                        title='PCA',
                        labels={'x': 'Component 1', 'y': 'Component 2'}
                    )
                    fig_pca.update_traces(marker=dict(size=5, line=dict(width=2, color='DarkSlateGrey')))
                    fig_pca.update_layout(showlegend=False)

                    fig_svd = px.scatter(
                        x=X_reduced_svd[:, 0], y=X_reduced_svd[:, 1],
                        color=[color_map[label] for label in y],
                        title='Truncated SVD',
                        labels={'x': 'Component 1', 'y': 'Component 2'}
                    )
                    fig_svd.update_traces(marker=dict(size=5, line=dict(width=2, color='DarkSlateGrey')))
                    fig_svd.update_layout(showlegend=False)

                    st.plotly_chart(fig_tsne)
                    st.plotly_chart(fig_pca)
                    st.plotly_chart(fig_svd)

            st.divider()

            st.header('Classifier :grey[Setup]')
            with st.container(height=850):
                with st.spinner('Generating...'):
                    st.info(':blue[Hint:] Select Test size')
                    st.warning(':orange[Warning:] Test size may change expected analysis.')

                    X = new_CleanedNull.drop(fraud_class, axis=1)
                    y = new_CleanedNull[fraud_class]

                    classificationSlider = st.slider("**Test size**", 0.1, 0.5)
                    st.toast(':orange[Test Size Modified...]')
                    st.write('##')

                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=classificationSlider,
                                                                        random_state=42)

                    X_train = X_train.values
                    X_test = X_test.values
                    y_train = y_train.values
                    y_test = y_test.values

                    classifiers = {
                        "LogisiticRegression": LogisticRegression(),
                        "KNearest": KNeighborsClassifier(),
                        "Support Vector Classifier": SVC(),
                        "DecisionTreeClassifier": DecisionTreeClassifier()
                    }

                    for key, classifier in classifiers.items():
                        classifier.fit(X_train, y_train)
                        training_score = cross_val_score(classifier, X_train, y_train, cv=5)
                        st.write(classifier.__class__.__name__, "training score: ",
                                 round(training_score.mean(), 2) * 100, "% accuracy score")

                ###################
                st.divider()
                st.subheader('Classifier Cross Validation')
                with st.spinner('Generating...'):

                    log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

                    grid_log_reg = GridSearchCV(LogisticRegression(), log_reg_params)
                    grid_log_reg.fit(X_train, y_train)

                    log_reg = grid_log_reg.best_estimator_

                    knears_params = {"n_neighbors": list(range(2, 5, 1)),
                                     'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}

                    grid_knears = GridSearchCV(KNeighborsClassifier(), knears_params)
                    grid_knears.fit(X_train, y_train)

                    knears_neighbors = grid_knears.best_estimator_

                    svc_params = {'C': [0.5, 0.7, 0.9, 1], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}
                    grid_svc = GridSearchCV(SVC(), svc_params)
                    grid_svc.fit(X_train, y_train)

                    svc = grid_svc.best_estimator_

                    tree_params = {"criterion": ["gini", "entropy"], "max_depth": list(range(2, 4, 1)),
                                   "min_samples_leaf": list(range(5, 7, 1))}
                    grid_tree = GridSearchCV(DecisionTreeClassifier(), tree_params)
                    grid_tree.fit(X_train, y_train)

                    tree_clf = grid_tree.best_estimator_

                    log_reg_score = cross_val_score(log_reg, X_train, y_train, cv=5)
                    st.write('Logistic Regression Cross Validation Score: ',
                             round(log_reg_score.mean() * 100, 2).astype(str) + '%')

                    knears_score = cross_val_score(knears_neighbors, X_train, y_train, cv=5)
                    st.write('Knears Neighbors Cross Validation Score',
                             round(knears_score.mean() * 100, 2).astype(str) + '%')

                    svc_score = cross_val_score(svc, X_train, y_train, cv=5)
                    st.write('Support Vector Classifier Cross Validation Score',
                             round(svc_score.mean() * 100, 2).astype(str) + '%')

                    tree_score = cross_val_score(tree_clf, X_train, y_train, cv=5)
                    st.write('DecisionTree Classifier Cross Validation Score',
                             round(tree_score.mean() * 100, 2).astype(str) + '%')

                    ################

                    undersample_X = CleanedNull.drop(fraud_class, axis=1)
                    undersample_y = CleanedNull[fraud_class]

                    for train_index, test_index in sss.split(undersample_X, undersample_y):
                        undersample_Xtrain, undersample_Xtest = undersample_X.iloc[train_index], undersample_X.iloc[
                            test_index]
                        undersample_ytrain, undersample_ytest = undersample_y.iloc[train_index], undersample_y.iloc[
                            test_index]

                    undersample_Xtrain = undersample_Xtrain.values
                    undersample_Xtest = undersample_Xtest.values
                    undersample_ytrain = undersample_ytrain.values
                    undersample_ytest = undersample_ytest.values

                    undersample_accuracy = []
                    undersample_precision = []
                    undersample_recall = []
                    undersample_f1 = []
                    undersample_auc = []

                    X_nearmiss, y_nearmiss = NearMiss().fit_resample(undersample_X.values, undersample_y.values)
                    st.write('NearMiss Label Distribution: {}'.format(Counter(y_nearmiss)))

                    for train, test in sss.split(undersample_Xtrain, undersample_ytrain):
                        undersample_pipeline = imbalanced_make_pipeline(NearMiss(sampling_strategy='majority'),
                                                                        log_reg)
                        undersample_model = undersample_pipeline.fit(undersample_Xtrain[train],
                                                                     undersample_ytrain[train])
                        undersample_prediction = undersample_model.predict(undersample_Xtrain[test])

                        undersample_accuracy.append(
                            undersample_pipeline.score(original_Xtrain[test], original_ytrain[test]))
                        undersample_precision.append(precision_score(original_ytrain[test], undersample_prediction))
                        undersample_recall.append(recall_score(original_ytrain[test], undersample_prediction))
                        undersample_f1.append(f1_score(original_ytrain[test], undersample_prediction))
                    # undersample_auc.append(roc_auc_score(original_ytrain[test], undersample_prediction))

            st.write('##')

            with st.expander('Visualizing Classification Leaning Curve'):
                st.subheader(f':green[Visualizing Classification Leaning Curve]')
                with st.spinner('Generating...'):
                    st.write('##')

                    def plot_learning_curve(estimator, X, y, title, cv, n_jobs, train_sizes):
                        train_sizes, train_scores, test_scores = learning_curve(
                            estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

                        train_scores_mean = np.mean(train_scores, axis=1)
                        train_scores_std = np.std(train_scores, axis=1)
                        test_scores_mean = np.mean(test_scores, axis=1)
                        test_scores_std = np.std(test_scores, axis=1)

                        fig = go.Figure()

                        fig.add_trace(go.Scatter(
                            x=train_sizes, y=train_scores_mean,
                            mode='lines+markers',
                            name='Training score',
                            line=dict(color='royalblue'),
                            error_y=dict(
                                type='data',
                                array=train_scores_std,
                                visible=True
                            )
                        ))

                        fig.add_trace(go.Scatter(
                            x=train_sizes, y=test_scores_mean,
                            mode='lines+markers',
                            name='Cross-validation score',
                            line=dict(color='orange'),
                            error_y=dict(
                                type='data',
                                array=test_scores_std,
                                visible=True
                            )
                        ))

                        fig.update_layout(
                            title=title,
                            xaxis_title='Training size (m)',
                            yaxis_title='Score',
                            legend=dict(x=0.5, y=-0.2, xanchor='center', yanchor='top'),
                            margin=dict(l=0, r=0, t=40, b=40)
                        )

                        return fig

                    estimators = {
                        "Logistic Regression": LogisticRegression(),
                        "K-Neighbors Classifier": KNeighborsClassifier(),
                        "Support Vector Classifier": SVC(),
                        "Decision Tree Classifier": DecisionTreeClassifier()
                    }

                    selected_estimators = list(estimators.keys())

                    train_sizes = np.linspace(0.1, 1.0, 5)
                    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
                    n_jobs = 1

                    for est_name in selected_estimators:
                        fig = plot_learning_curve(estimators[est_name], X, y, f"{est_name} Learning Curve", cv, n_jobs,
                                                  train_sizes)
                        st.plotly_chart(fig)

            st.divider()

            st.header(':orange[ROC Score]')
            with st.expander("See :red[Documentation]", expanded=False):
                st.subheader(':blue[Cross Validation]')
                st.help(sklearn.model_selection._validation)
                st.divider()
                st.subheader(':blue[ROC AUC Score]')
                st.help(sklearn.metrics._ranking)

            with st.container(height=200):
                with st.spinner('Generating...'):
                    log_reg_pred = cross_val_predict(log_reg, X_train, y_train, cv=5,
                                                     method="decision_function")

                    knears_pred = cross_val_predict(knears_neighbors, X_train, y_train, cv=5)

                    svc_pred = cross_val_predict(svc, X_train, y_train, cv=5,
                                                 method="decision_function")

                    tree_pred = cross_val_predict(tree_clf, X_train, y_train, cv=5)

                    st.write('Logistic Regression: ', roc_auc_score(y_train, log_reg_pred))
                    st.write('KNears Neighbors: ', roc_auc_score(y_train, knears_pred))
                    st.write('Support Vector Classifier: ', roc_auc_score(y_train, svc_pred))
                    st.write('Decision Tree Classifier: ', roc_auc_score(y_train, tree_pred))

            st.write('##')

            with st.expander('Visualizing ROC Score'):
                st.subheader(f':green[Visualizing ROC Score]')
                with st.spinner('Generating...'):
                    st.write('##')

                    log_fpr, log_tpr, log_thresold = roc_curve(y_train, log_reg_pred)
                    knear_fpr, knear_tpr, knear_threshold = roc_curve(y_train, knears_pred)
                    svc_fpr, svc_tpr, svc_threshold = roc_curve(y_train, svc_pred)
                    tree_fpr, tree_tpr, tree_threshold = roc_curve(y_train, tree_pred)

                    def graph_roc_curve_multiple(log_fpr, log_tpr, knear_fpr, knear_tpr, svc_fpr, svc_tpr, tree_fpr,
                                                 tree_tpr):
                        fig = go.Figure()

                        fig.add_trace(go.Scatter(x=log_fpr, y=log_tpr,
                                                 mode='lines',
                                                 name='Logistic Regression Classifier Score: {:.4f}'.format(
                                                     roc_auc_score(y_train, log_reg_pred))))
                        fig.add_trace(go.Scatter(x=knear_fpr, y=knear_tpr,
                                                 mode='lines', name='KNears Neighbors Classifier Score: {:.4f}'.format(
                                roc_auc_score(y_train, knears_pred))))
                        fig.add_trace(go.Scatter(x=svc_fpr, y=svc_tpr,
                                                 mode='lines', name='Support Vector Classifier Score: {:.4f}'.format(
                                roc_auc_score(y_train, svc_pred))))
                        fig.add_trace(go.Scatter(x=tree_fpr, y=tree_tpr,
                                                 mode='lines', name='Decision Tree Classifier Score: {:.4f}'.format(
                                roc_auc_score(y_train, tree_pred))))

                        fig.add_trace(
                            go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), showlegend=False))

                        fig.update_layout(title='ROC Curve \n Top 4 Classifiers',
                                          xaxis_title='False Positive Rate',
                                          yaxis_title='True Positive Rate',
                                          annotations=[dict(x=0.5, y=0.5, xref='x', yref='y',
                                                            text='Minimum ROC Score of 50% \n (This is the minimum score to get)',
                                                            showarrow=True, arrowhead=7, ax=0, ay=-40)])

                        return fig

                    fig = graph_roc_curve_multiple(log_fpr, log_tpr, knear_fpr, knear_tpr, svc_fpr, svc_tpr, tree_fpr,
                                                   tree_tpr)
                    st.plotly_chart(fig)

            st.write('##')

            with st.expander('Visualizing Logistic Regression ROC Curve'):
                st.subheader(f':green[Visualizing Logistic Regression ROC Curve]')
                with st.spinner('Generating...'):
                    st.write('##')

                    def logistic_roc_curve(log_fpr, log_tpr):
                        fig = go.Figure()

                        fig.add_trace(
                            go.Scatter(x=log_fpr, y=log_tpr, mode='lines', name='ROC Curve',
                                       line=dict(color='blue', width=2)))

                        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Guess',
                                                 line=dict(color='red', dash='dash')))

                        fig.update_layout(
                            title='Logistic Regression ROC Curve',
                            xaxis=dict(title='False Positive Rate', range=[-0.01, 1]),
                            yaxis=dict(title='True Positive Rate', range=[0, 1]),
                            width=800,
                            height=600
                        )
                        return fig

                    fig = logistic_roc_curve(log_fpr, log_tpr)

                    st.plotly_chart(fig)

            st.divider()

            st.header(':orange[Fitting]')
            with st.container(height=600):
                with st.spinner('Generating...'):
                    precision, recall, threshold = precision_recall_curve(y_train, log_reg_pred)

                    y_pred = log_reg.predict(X_train)

                    st.subheader('Overfitting')
                    st.write('Recall Score: {:.2f}'.format(recall_score(y_train, y_pred)))
                    st.write('Precision Score: {:.2f}'.format(precision_score(y_train, y_pred)))
                    st.write('F1 Score: {:.2f}'.format(f1_score(y_train, y_pred)))
                    st.write('Accuracy Score: {:.2f}'.format(accuracy_score(y_train, y_pred)))

                    st.subheader('Actual Fit')
                    st.write("Accuracy Score: {:.2f}".format(np.mean(undersample_accuracy)))
                    st.write("Precision Score: {:.2f}".format(np.mean(undersample_precision)))
                    st.write("Recall Score: {:.2f}".format(np.mean(undersample_recall)))
                    st.write("F1 Score: {:.2f}".format(np.mean(undersample_f1)))

                    st.divider()

                    undersample_y_score = log_reg.decision_function(original_Xtest)

                    undersample_average_precision = average_precision_score(original_ytest, undersample_y_score)

                    st.write('Average precision-recall score: {0:0.2f}'.format(
                        undersample_average_precision))

            with st.expander('Visualizing Precision-Recall Curve'):
                st.subheader(f':green[Visualizing Precision-Recall Curve]')
                with st.spinner('Generating...'):
                    st.write('##')

                    trace = go.Scatter(
                        x=recall,
                        y=precision,
                        mode='lines+markers',
                        fill='tozeroy',
                        fillcolor='rgba(72, 166, 255, 0.2)',
                        line=dict(color='rgba(0, 74, 147, 0.6)'),
                        name='Precision-Recall curve'
                    )

                    layout = go.Layout(
                        title='UnderSampling Precision-Recall curve: <br> Average Precision-Recall Score = {0:0.2f}'.format(
                            undersample_average_precision),
                        xaxis=dict(title='Recall'),
                        yaxis=dict(title='Precision'),
                        yaxis_range=[0.0, 1.05],
                        xaxis_range=[0.0, 1.0],
                        font=dict(size=16)
                    )

                    fig = go.Figure(data=[trace], layout=layout)

                    st.plotly_chart(fig)

            st.divider()

            st.header(':orange[Accuracy Score]')
            with st.expander("See :red[Documentation]", expanded=False):
                st.subheader(':blue[RandomizedSearchCV]')
                st.help(sklearn.model_selection._search)

            with st.container(height=650):
                with st.spinner('Generating...'):
                    st.write(
                        'Length of X (train): {} | Length of y (train): {}'.format(len(original_Xtrain),
                                                                                   len(original_ytrain)))
                    st.write(
                        'Length of X (test): {} | Length of y (test): {}'.format(len(original_Xtest),
                                                                                 len(original_ytest)))

                    accuracy_lst = []
                    precision_lst = []
                    recall_lst = []
                    f1_lst = []
                    auc_lst = []

                    log_reg_sm = LogisticRegression()

                    rand_log_reg = RandomizedSearchCV(LogisticRegression(), log_reg_params, n_iter=4)

                    log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
                    for train, test in sss.split(original_Xtrain, original_ytrain):
                        pipeline = imbalanced_make_pipeline(SMOTE(sampling_strategy='minority'), rand_log_reg)
                        model = pipeline.fit(original_Xtrain[train], original_ytrain[train])
                        best_est = rand_log_reg.best_estimator_
                        prediction = best_est.predict(original_Xtrain[test])

                        accuracy_lst.append(pipeline.score(original_Xtrain[test], original_ytrain[test]))
                        precision_lst.append(precision_score(original_ytrain[test], prediction))
                        recall_lst.append(recall_score(original_ytrain[test], prediction))
                        f1_lst.append(f1_score(original_ytrain[test], prediction))
                        auc_lst.append(roc_auc_score(original_ytrain[test], prediction))

                    st.write('')
                    st.write("accuracy: {}".format(np.mean(accuracy_lst)))
                    st.write("precision: {}".format(np.mean(precision_lst)))
                    st.write("recall: {}".format(np.mean(recall_lst)))
                    st.write("f1: {}".format(np.mean(f1_lst)))

                    ###############
                    labels = ['No Fraud', 'Fraud']
                    smote_prediction = best_est.predict(original_Xtest)
                    st.write(classification_report(original_ytest, smote_prediction, target_names=labels))

                    y_score = best_est.decision_function(original_Xtest)

                    average_precision = average_precision_score(original_ytest, y_score)

                    st.write('Average precision-recall score: {0:0.2f}'.format(
                        average_precision))

            st.write('##')

            with st.expander('Visualizing Precision-Recall Curve'):
                st.subheader(f':green[Visualizing Precision-Recall Curve]')
                with st.spinner('Generating...'):
                    st.write('##')

                    precision, recall, _ = precision_recall_curve(original_ytest, y_score)

                    fig = go.Figure()

                    fig.add_trace(go.Scatter(
                        x=recall,
                        y=precision,
                        mode='lines+markers',
                        name='Precision-Recall curve',
                        line=dict(color='red'),
                        fill='tozeroy',
                        fillcolor='rgba(245, 155, 0, 0.2)'
                    ))

                    fig.update_layout(
                        title='OverSampling Precision-Recall curve: <br> Average Precision-Recall Score ={0:0.2f}'.format(
                            average_precision),
                        xaxis_title='Recall',
                        yaxis_title='Precision',
                        xaxis=dict(range=[0, 1]),
                        yaxis=dict(range=[0, 1.05]),
                        width=800,
                        height=400,
                        font=dict(size=16)
                    )

                    st.plotly_chart(fig)

                    sm = SMOTE(sampling_strategy='minority', random_state=42)

                    Xsm_train, ysm_train = sm.fit_resample(original_Xtrain, original_ytrain)

                    t0 = time.time()
                    log_reg_sm = grid_log_reg.best_estimator_
                    log_reg_sm.fit(Xsm_train, ysm_train)
                    t1 = time.time()
                    st.write("Fitting oversample data took :{} sec".format(t1 - t0))

            st.write('##')

            with st.expander('Visualizing Confusion Matrix'):
                st.subheader(f':green[Visualizing Confusion Matrix]')
                with st.spinner('Generating...'):
                    st.write('##')

                    y_pred_log_reg = log_reg_sm.predict(X_test)

                    y_pred_knear = knears_neighbors.predict(X_test)
                    y_pred_svc = svc.predict(X_test)
                    y_pred_tree = tree_clf.predict(X_test)

                    log_reg_cf = confusion_matrix(y_test, y_pred_log_reg)
                    kneighbors_cf = confusion_matrix(y_test, y_pred_knear)
                    svc_cf = confusion_matrix(y_test, y_pred_svc)
                    tree_cf = confusion_matrix(y_test, y_pred_tree)

                    confusion_matrices = {
                        "Logistic Regression": log_reg_cf,
                        "KNeighbors Classifier": kneighbors_cf,
                        "Support Vector Classifier": svc_cf,
                        "Decision Tree Classifier": tree_cf
                    }

                    figures = {}

                    for title, cf_matrix in confusion_matrices.items():
                        df_cm = pd.DataFrame(cf_matrix, index=[i for i in "01"], columns=[i for i in "01"])
                        fig = px.imshow(df_cm, text_auto=True, color_continuous_scale=colors)
                        fig.update_layout(title=title + " Confusion Matrix",
                                          xaxis_title="Predicted Label",
                                          yaxis_title="True Label")
                        figures[title] = fig

                    for title, fig in figures.items():
                        st.plotly_chart(fig)

            st.write('##')

            st.header(':orange[Score]')
            with st.container(height=1300):
                with st.spinner('Generating...'):
                    st.subheader(':red[***Logistic Regression***]')
                    st.write(classification_report(y_test, y_pred_log_reg))

                    st.subheader(':red[***KNears Neighbors***]')
                    st.write(classification_report(y_test, y_pred_knear))

                    st.subheader(':red[***Support Vector Classifier***]')
                    st.write(classification_report(y_test, y_pred_svc))

                    st.subheader(':red[***Support Vector Classifier***]')
                    st.write(classification_report(y_test, y_pred_tree))

                    ###################

                    y_pred = log_reg.predict(X_test)
                    undersample_score = accuracy_score(y_test, y_pred)

                    y_pred_sm = best_est.predict(original_Xtest)
                    oversample_score = accuracy_score(original_ytest, y_pred_sm)

                    d = {'Technique': ['Random UnderSampling', 'Oversampling (SMOTE)'],
                         'Score': [undersample_score, oversample_score]}
                    final_df = pd.DataFrame(data=d)

                    score = final_df['Score']
                    final_df.drop('Score', axis=1, inplace=True)
                    final_df.insert(1, 'Score', score)

                    st.write(final_df)

            st.write('##')

            with st.expander('Visualising Confusion Matrix (Under Sample)'):
                st.subheader(f':green[Visualising Confusion Matrix (Under Sample)]')
                with st.spinner('Generating...'):
                    st.write('##')

                    n_inputs = X_train.shape[1]

                    undersample_model = Sequential([
                        Dense(n_inputs, input_shape=(n_inputs,), activation='relu'),
                        Dense(32, activation='relu'),
                        Dense(2, activation='softmax')
                    ])

                    undersample_model.summary()

                    undersample_model.compile(Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy',
                                              metrics=['accuracy'])

                    undersample_model.fit(X_train, y_train, validation_split=0.2, batch_size=25, epochs=20,
                                          shuffle=True,
                                          verbose=2)

                    undersample_predictions = undersample_model.predict(original_Xtest, batch_size=200, verbose=0)
                    undersample_fraud_predictions = np.argmax(undersample_predictions, axis=1)

                    def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap='Blues'):
                        trace = go.Heatmap(z=cm, x=classes, y=classes, colorscale=cmap)
                        layout = go.Layout(title=title, xaxis=dict(title='Predicted label'),
                                           yaxis=dict(title='True label'))
                        fig = go.Figure(data=[trace], layout=layout)
                        return fig

                    undersample_cm = confusion_matrix(original_ytest, undersample_fraud_predictions)
                    actual_cm = confusion_matrix(original_ytest, original_ytest)
                    labels = ['No Fraud', 'Fraud']

                    st.subheader('Random UnderSample Confusion Matrix')
                    undersample_fig = plot_confusion_matrix(undersample_cm, labels,
                                                            title="Random UnderSample Confusion Matrix",
                                                            cmap='Reds')
                    st.plotly_chart(undersample_fig)

                    st.subheader('Confusion Matrix (with 100% accuracy)')
                    actual_fig = plot_confusion_matrix(actual_cm, labels, title="Confusion Matrix (with 100% accuracy)",
                                                       cmap='Greens')
                    st.plotly_chart(actual_fig)

            st.write('##')

            with st.expander('Visualising Confusion Matrix (Over Sample)'):
                st.subheader(f':green[Visualising Confusion Matrix (Over Sample)]')
                with st.spinner('Generating...'):
                    st.write('##')

                    n_inputs = Xsm_train.shape[1]

                    oversample_model = Sequential([
                        Dense(n_inputs, input_shape=(n_inputs,), activation='relu'),
                        Dense(32, activation='relu'),
                        Dense(2, activation='softmax')
                    ])
                    oversample_model.compile(Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy',
                                             metrics=['accuracy'])
                    oversample_model.fit(Xsm_train, ysm_train, validation_split=0.2, batch_size=300, epochs=20,
                                         shuffle=True,
                                         verbose=2)

                    oversample_predictions = oversample_model.predict(original_Xtest, batch_size=200, verbose=0)
                    oversample_fraud_predictions = oversample_model.predict(original_Xtest, batch_size=200, verbose=0)
                    oversample_smote = confusion_matrix(original_ytest, oversample_fraud_predictions)
                    actual_cm = confusion_matrix(original_ytest, original_ytest)
                    labels = ['No Fraud', 'Fraud']

                    def create_confusion_matrix_figure(matrix, labels, title, colorscale):
                        z = matrix.tolist()
                        z_text = [[str(y) for y in x] for x in z]
                        fig = ff.create_annotated_heatmap(z, x=labels, y=labels, annotation_text=z_text,
                                                          colorscale=colorscale)
                        fig.update_layout(title_text=title, margin=dict(t=50, l=200))
                        return fig

                    fig1 = create_confusion_matrix_figure(oversample_smote, labels,
                                                          "OverSample (SMOTE) \n Confusion Matrix",
                                                          colorscale='Oranges')
                    fig2 = create_confusion_matrix_figure(actual_cm, labels, "Confusion Matrix \n (with 100% accuracy)",
                                                          colorscale='Greens')

                    st.plotly_chart(fig1)
                    st.plotly_chart(fig2)


    if __name__ == "__main__":
        main()

    st.toast(':green[Ready!]')
