import time

import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import seaborn as sns
import sklearn.ensemble
import streamlit as st
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

with st.spinner('Loading...'):
    st.toast('Checking Environment')


    def main():
        titleCol1, titleCol2, titleCol3 = st.columns([1, 1, 1])
        with titleCol1:
            pass
        with titleCol2:
            st.title('***Data Analysis*** 🔎')
            st.write(':grey[Do data analysis using toggles]')
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
            with open("csv/titanic_sample.csv", "rb") as file:
                data = file.read()
                st.download_button(
                    label="Download :red[Dataset]",
                    data=data,
                    file_name="titanic_sample.csv",
                    mime="text/csv"
                )

                st.subheader(
                    'The dataset appears to be a sample from the Titanic passenger list. Here are the columns included in the dataset.')
                st.write(':orange[***PassengerId***]: A unique identifier for each passenger.')
                st.write(':orange[***Survived***]: Survival status (0 = No, 1 = Yes).')
                st.write(':orange[***Pclass***]: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd).')
                st.write(':orange[***Name***]: The name of the passenger.')
                st.write(':orange[***Sex***]: Gender of the passenger.')
                st.write(':orange[***Age***]: Age of the passenger.')
                st.write(':orange[***SibSp***]:  Number of siblings or spouses aboard the Titanic.')
                st.write(':orange[***Parch***]: Number of parents or children aboard the Titanic.')
                st.write(':orange[***Ticket***]: Ticket number.')
                st.write(':orange[***Fare***]: Passenger fare.')
                st.write(':orange[***Cabin***]:  Cabin number.')
                st.write(
                    ':orange[***Embarked***]: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).')
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
                    ':blue[***Step 3***]: Drop label[Name, Sex, Ticket, Fare, Cabin, Embarked]. [Info]: Labels which contains String values. [:red[Disclaimer]: You may have to perform label drop for differnt datasets]')
                st.write(
                    ':blue[***Step 4***]: Select a suitable label in Random Forest Classification area and Run the Test by sliding the test size bar. [:red[Disclaimer]: Test size may vary in different datasets]')
                st.write(
                    ':blue[***Step 5***]: Select a suitable label in Bar Plot. [:red[Disclaimer]: Label may vary in different datasets]')
                st.write(
                    ':blue[***Step 6***]: Drop non-correlated label in Heatmap :grey[ Correlation]. [:red[Disclaimer]: Label may vary in different datasets]')
                st.write(
                    ':blue[***Step 7***]: Visualize the Interactive Plots, [Info]: Pinch Out to Zoom IN, Pinch In to Zoom Out. OR Select an area to Zoom. Use auto-scale to revert ')

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
                                container_del.download_button("Download :red[CSV]", delete_df_drop.to_csv(index=False),
                                                              file_name="modified_csv.csv", key=0)

                        if selected_indices:
                            st.subheader(":green[Result]")
                            st.write(delete_df_drop)

        if uploaded_file is not None:
            st.divider()
            # OverView #
            st.title("***Over:grey[view]***")

            with st.expander("See :red[Documentation]", expanded=False):

                st.subheader(':blue[Describe]')
                st.help(pd.DataFrame.describe)
                st.divider()

            with st.container(height=600, border=True):
                with st.spinner('Generating...'):
                    tab_main1, tab_main2 = st.tabs(["Dataframe", "Describe"])

                    with tab_main1:
                        st.subheader("Dataframe")
                        st.write(df_original)

                    with tab_main2:
                        st.subheader("Description")
                        st.write(df_original.describe())

            st.divider()

            # Data Manipulation #
            st.title(":orange[**Data Manipulation**]")

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

            text_to_float = "Try to avoid string literals"

            st.markdown(f'<div class="float-text">{text_to_float}</div>', unsafe_allow_html=True)

            # Data dropping #
            st.write('##')
            with st.expander("See :red[Documentation]", expanded=False):

                st.subheader(':blue[Drop Na]')
                st.help(pd.DataFrame.dropna)
                st.divider()
                st.subheader(':blue[Drop Duplicates]')
                st.help(pd.DataFrame.drop_duplicates)

            st.subheader("Drop Column")
            with st.container(height=1000, border=True):
                with st.spinner('Generating...'):

                    dropColumn = st.multiselect("**Choose label to drop**", df_original.columns)

                    st.toast(':red[Dropping Columns...]')

                    chooseDropFill = st.radio("**Missing value strategy**",
                                              ["Drop rows having missing values",
                                               "Fill missing values with mean [:orange[Avoid string]]"])
                    dropBarData = df_original.drop(columns=dropColumn)

                    CleanedNull = None
                    if chooseDropFill == "Drop rows having missing values":
                        CleanedNull = dropBarData.dropna().drop_duplicates()

                        st.toast(':red[Removing Null...]')

                        st.info(":grey[Null value :red[Removed]]")
                    elif chooseDropFill == "Fill missing values with mean":
                        CleanedNull = dropBarData.fillna(dropBarData.mean())

                        st.toast(':orange[Filling mean...]')

                        st.success(":grey[Null value filled with :green[Mean]]")

                # modified overview #

                with st.container(height=700, border=True):
                    with st.spinner('Generating...'):
                        st.header("***Over:grey[view] :blue[Simulation]***")
                        st.write(":grey[This process is real time]")

                        # tabular form #
                        tab1, tab2 = st.tabs(["Dataframe", "Describe"])

                        with tab1:
                            st.subheader("Dataframe")
                            st.write(CleanedNull)

                        with tab2:
                            st.subheader("Description")
                            st.write(CleanedNull.describe())

            # -----Data Classification
            st.write('##')

            st.subheader("Random Forest Classification")
            with st.expander("See :red[Documentation]", expanded=False):

                st.subheader(':blue[Random Forest Regression]')
                st.help(sklearn.ensemble.RandomForestRegressor)
                st.divider()
                st.subheader(':blue[Train Test Split]')
                st.help(sklearn.model_selection.train_test_split)
                st.divider()
                st.subheader(':blue[Train Test Split]')
                st.help(sklearn.metrics._classification)

            with st.container(height=600, border=True):
                with st.spinner('Generating...'):
                    st.warning(":orange[Disclaimer:] Do not perform on string")

                    st.write('##')
                    classificationColumn = st.selectbox("**Select target label**", dropBarData.columns)

                    classificationSlider = st.slider("**Test size**", 0.1, 0.3, 0.5)

                    st.toast(':orange[Test Size Modified...]')
                    if st.button(":red[Run]"):
                        X = CleanedNull.drop(columns=[classificationColumn])
                        y = CleanedNull[classificationColumn]
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=classificationSlider,
                                                                            random_state=42)
                        clf = RandomForestClassifier()
                        clf.fit(X_train, y_train)
                        y_pred = clf.predict(X_test)
                        accuracy = accuracy_score(y_test, y_pred)

                        st.success(f"### :grey[Accuracy:] :green[{accuracy}]")

            # -----Bar Plot
            st.write('##')

            st.subheader("Bar Plot")
            with st.expander("See :red[Documentation]", expanded=False):
                st.subheader(':blue[Bar Plot]')
                st.help(go.Bar)
                st.divider()

            with st.container(height=700, border=True):
                with st.spinner('Generating...'):
                    label = st.selectbox("**Select plot label**", CleanedNull.columns)
                    with st.container(height=550, border=True):
                        st.write('Unique values: {}'.format(CleanedNull[label].nunique()))
                        cnt_srs = CleanedNull[label].value_counts(normalize=True)
                        trace = go.Bar(
                            x=cnt_srs.index,
                            y=cnt_srs.values,
                            marker=dict(
                                color='#FF66CC',
                            ),
                        )
                        layout = go.Layout(
                            font=dict(size=14),
                            width=800,
                            height=500,
                        )
                        data = [trace]
                        fig = go.Figure(data=data, layout=layout)
                        st.plotly_chart(fig, use_container_width=True)

            # -------Heatmap
            st.write('##')

            st.subheader("Heatmap :grey[ Correlation]")
            with st.expander("See :red[Documentation]", expanded=False):
                st.subheader(':blue[Heatmap]')
                st.help(px.imshow)
                st.divider()
            with st.container(height=700, border=True):
                with st.spinner('Generating...'):
                    st.warning(":orange[Disclaimer:] Do not perform on string")

                    label_heatmap = st.multiselect("**Select heatmap drop label**", CleanedNull.columns)
                    with st.container(height=500, border=True):
                        dropHeatmap = CleanedNull.drop(columns=label_heatmap)
                        corr_matrix = dropHeatmap.corr()
                        fig, axes = plt.subplots(figsize=(21, 7))
                        sns.heatmap(data=corr_matrix, annot=True, cmap='crest')
                        plt.tight_layout()
                        st.pyplot(fig)


    if __name__ == "__main__":
        main()

    st.toast(':green[Ready!]')
