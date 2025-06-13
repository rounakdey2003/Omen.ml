import csv
from io import StringIO

import streamlit as st


def main():
    st.title(":orange[CSV] to :red[Text] Converter")

    uploaded_file = st.file_uploader("8) Upload CSV File", type=["csv"])
    if uploaded_file is not None:
        st.success('Data read :green[**Successfully**]')
    else:
        st.error("Please import any :red['**.CSV**'] file to start")

    if st.button("8) Convert to :red[Text]", disabled=not uploaded_file):
        st.toast(':orange[Converting...]')
        text_data = convert_to_text(uploaded_file)
        st.success("Text file created :green[successfully]")
        st.download_button(
            label="Download :red[Text]",
            data=text_data,
            file_name="output.txt",
            mime="text/plain"
        )


def convert_to_text(uploaded_file):
    text_output = StringIO()

    reader = csv.reader(uploaded_file, delimiter=',', quotechar='"')
    for row in reader:
        for item in row:
            text_output.write(item + "\t")
        text_output.write("\n")
    return text_output.getvalue()


if __name__ == "__main__":
    main()
