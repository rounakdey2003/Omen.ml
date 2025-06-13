import streamlit as st

with st.spinner('Loading...'):
    st.toast('Checking Environment')


    def main():
        titleCol1, titleCol2, titleCol3 = st.columns([1, 1, 1])
        with titleCol1:
            pass
        with titleCol2:
            st.title('***Python GPT*** 🐍')
            st.write(':grey[Get python module documentations]')
        with titleCol3:
            pass

        st.divider()

        st.title(":blue[Search]")

        ai_column1, ai_column2 = st.columns([3, 1, ])

        if "prompt_history" not in st.session_state:
            st.session_state.prompt_history = []

        with ai_column1:

            with st.container(height=500, border=True):

                response_suggest = None
                result_1 = None
                result_2 = None

                expand_column1, expand_column2 = st.columns([1, 1])
                with expand_column1:
                    with st.expander("Pandas Suggestions"):

                        if st.button("Dataframe"):
                            prompt = 'pandas.Dataframe'
                            if prompt:
                                st.session_state.prompt_history.append(prompt)
                                result_2 = ":green[Result]:"
                                response_suggest = prompt

                        if st.button("Index"):
                            prompt = "pandas.Dataframe.index"
                            if prompt:
                                st.session_state.prompt_history.append(prompt)
                                result_2 = ":green[Result]:"
                                response_suggest = prompt

                        if st.button("Copy"):
                            prompt = "pandas.Dataframe.copy"
                            if prompt:
                                st.session_state.prompt_history.append(prompt)
                                result_2 = ":green[Result]:"
                                response_suggest = prompt

                        if st.button("Mean"):
                            prompt = "pandas.Dataframe.mean"
                            if prompt:
                                st.session_state.prompt_history.append(prompt)
                                result_2 = ":green[Result]:"
                                response_suggest = prompt

                with expand_column2:
                    with st.expander("Numpy Suggestions"):
                        if st.button("Polynomial"):
                            prompt = "numpy.polynomial"
                            if prompt:
                                st.session_state.prompt_history.append(prompt)
                                result_2 = ":green[Result]:"
                                response_suggest = prompt

                        if st.button("EMath"):
                            prompt = "numpy.emath"
                            if prompt:
                                st.session_state.prompt_history.append(prompt)
                                result_2 = ":green[Result]:"
                                response_suggest = prompt

                        if st.button("Tan"):
                            prompt = "numpy.tan"
                            if prompt:
                                st.session_state.prompt_history.append(prompt)
                                result_2 = ":green[Result]:"
                                response_suggest = prompt

                        if st.button("Power"):
                            prompt = "numpy.pow"
                            if prompt:
                                st.session_state.prompt_history.append(prompt)
                                result_2 = ":green[Result]:"
                                response_suggest = prompt

                expand_column3, expand_column4 = st.columns([1, 1])
                with expand_column3:
                    with st.expander("Streamlit Suggestions"):
                        if st.button("Write"):
                            prompt = "streamlit.write"
                            if prompt:
                                st.session_state.prompt_history.append(prompt)
                                result_2 = ":green[Result]:"
                                response_suggest = prompt

                        if st.button("Header"):
                            prompt = "streamlit.header"
                            if prompt:
                                st.session_state.prompt_history.append(prompt)
                                result_2 = ":green[Result]:"
                                response_suggest = prompt

                        if st.button("Info"):
                            prompt = "streamlit.info"
                            if prompt:
                                st.session_state.prompt_history.append(prompt)
                                result_2 = ":green[Result]:"
                                response_suggest = prompt

                        if st.button("Expander"):
                            prompt = "streamlit.expander"
                            if prompt:
                                st.session_state.prompt_history.append(prompt)
                                result_2 = ":green[Result]:"
                                response_suggest = prompt

                with expand_column4:
                    with st.expander("Plotly Suggestions"):
                        if st.button("Bar Plot"):
                            prompt = "plotly.express.bar"
                            if prompt:
                                st.session_state.prompt_history.append(prompt)
                                result_2 = ":green[Result]:"
                                response_suggest = prompt

                        if st.button("Map Box"):
                            prompt = "plotly.express.scatter_mapbox"
                            if prompt:
                                st.session_state.prompt_history.append(prompt)
                                result_2 = ":green[Result]:"
                                response_suggest = prompt

                        if st.button("Heatmap"):
                            prompt = "plotly.express.density_heatmap"
                            if prompt:
                                st.session_state.prompt_history.append(prompt)
                                result_2 = ":green[Result]:"
                                response_suggest = prompt

                        if st.button("Histogram"):
                            prompt = "plotly.express.histogram"
                            if prompt:
                                st.session_state.prompt_history.append(prompt)
                                result_2 = ":green[Result]:"
                                response_suggest = prompt

                prompt = st.text_area("Enter your module here:",
                                      help=':grey[**Example**]: pandas.Dataframe, numpy.polynomial, plotly.express.bar, streamlit.write')

                if st.button("Generate :red[Response]"):
                    if prompt:

                        response = prompt

                        result_1 = ":green[Result:]"

                        response_suggest = response

                    else:
                        st.warning("Please enter a prompt before generating a response.")

            with st.expander("See Result", expanded=True):
                if response_suggest is None:
                    st.error("You don't have any result")
                    st.info(
                        ':blue[**Try**]: pandas.Dataframe, numpy.polynomial, plotly.express.bar, streamlit.write, etc')
                    st.warning(':orange[**Avoid**]: pd.Dataframe, pandas.Dataframe(), etc ')
                    st.write(
                        ':red[Do not modify module name or use parenthesis in the end.] :blue[ Only use the actual name or path of the module.]')
                else:
                    st.write(result_2)
                    st.write(result_1)
                    st.help(response_suggest)
                    st.session_state.prompt_history.append(prompt)

        with ai_column2:
            with st.expander("Session :orange[History]"):
                if st.session_state.prompt_history:
                    for idx, query in enumerate(st.session_state.prompt_history, start=1):
                        st.write(f"{idx}. {query}")
                else:
                    st.info("No prompts entered yet.")


    if __name__ == "__main__":
        main()

    st.toast(':green[Ready!]')
