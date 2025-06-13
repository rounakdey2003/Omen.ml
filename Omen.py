import time

import streamlit as st
from PIL import Image

st.set_page_config(layout='wide', page_icon=Image.open('image/omenLogo.png'), initial_sidebar_state='expanded',
                   page_title='Omen', menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': ":rainbow[O]:grey[**men**] is a python web app which allows user to perform :red[Infinity]"})

with st.spinner('Loading...'):
    st.toast('Checking Environment')

    st.title('**:rainbow[O]MEN**:grey[.ml]')

    subtitle = '**Empower your creativity with the precision of ML.**'


    def stream_data():
        for word in subtitle.split(" "):
            yield word + " "
            time.sleep(0.1)


    if ("Stream data"):
        st.write_stream(stream_data)

    st.divider()

    searchCol1, searchCol2, searchCol3 = st.columns([1, 3, 1])

    with searchCol1:
        pass
    with searchCol2:
        search_query = st.text_input(label='', placeholder='Search for a page',
                                     help=':grey[**Keywords**] = PRO, Analyser, Analysis, GPT, Math, Tool, etc ')

        pages = [
            {"label": ":red-background[:red[Data Analysis]]", "page": "pages/Data Analysis.py",
             "help": "Do data analysis using toogles"},
            {"label": ":red-background[:red[Math GPT]]", "page": "pages/Math GPT.py",
             "help": "Solve basic-complex graphical problems"},
            {"label": ":red-background[:red[ToolKit]]", "page": "pages/ToolKit.py",
             "help": "Everyday converters for PDF, Text, Docx"},
            {"label": ":red-background[:orange[Lung Cancer Analyser (PRO)]]", "page": "pages/Lung Cancer Analyser (PRO).py",
             "help": "Analyse complex data for Lung Cancer disease"},
            {"label": ":red-background[:red[Disease Prediction]]", "page": "pages/Disease Prediction.py",
             "help": "Predict the disease from your report"},
            {"label": ":red-background[:orange[Credit Card Fraud Analyser (PRO)]]", "page": "pages/Credit Card Fraud Analyser (PRO).py",
             "help": "Analyse complex data for credit card fraud"},
            {"label": ":red-background[:red[Python GPT]]", "page": "pages/Python GPT.py",
             "help": "Get python module documentations"},
        ]

        if search_query:
            filtered_pages = [page for page in pages if search_query.lower() in page["label"].lower()]

            for page in filtered_pages:
                st.page_link(page=page["page"], label=page["label"], help=page["help"], use_container_width=False)

    with searchCol3:
        pass

    st.write('##')

    bodyCol1, bodyCol2 = st.columns([1, 1])

    with bodyCol1:
        with st.container(height=400, border=True):
            st.image('image/1.jpeg')
        if st.button(":orange[Lung Cancer Analyser (PRO)]"):
            st.switch_page("pages/Lung Cancer Analyser (PRO).py")

        st.write('##')

        with st.container(height=400, border=True):
            st.image('image/2.jpeg')
        if st.button(":red[Data Analysis]"):
            st.switch_page("pages/Data Analysis.py")

        st.write('##')

        with st.container(height=400, border=True):
            st.image('image/3.jpeg')
        if st.button(":red[Disease Prediction]"):
            st.switch_page("pages/Disease Prediction.py")

        st.write('##')

    with bodyCol2:
        with st.container(height=400, border=True):
            st.image('image/4.jpeg')
        if st.button(":orange[Credit Card Fraud Analyser (PRO)]"):
            st.switch_page("pages/Credit Card Fraud Analyser (PRO).py")

        st.write('##')

        with st.container(height=400, border=True):
            st.image('image/5.jpeg')
        if st.button(":red[Python GPT]"):
            st.switch_page("pages/Python GPT.py")

        st.write('##')

        with st.container(height=400, border=True):
            st.image('image/6.jpeg')
        if st.button(":red[Math GPT]"):
            st.switch_page("pages/Math GPT.py")

        st.write('##')

        with st.container(height=400, border=True):
            st.image('image/7.jpeg')
        if st.button(":red[ToolKit]"):
            st.switch_page("pages/ToolKit.py")

        st.write('##')

    st.divider()

st.page_link(page="https://github.com/rounakdey2003", label=":blue-background[:blue[Github]]",
                     help='Teleport to Github',
                     use_container_width=False)

st.toast(':green[Ready!]')
