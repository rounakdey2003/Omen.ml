import csv
from io import StringIO

import PyPDF2
import streamlit as st


def main():
    st.title(":orange[PDF] to :red[CSV] Converter")

    uploaded_file = st.file_uploader("7) Upload PDF File", type=["pdf"])
    if uploaded_file is not None:
        st.success('Data read :green[**Successfully**]')
    else:
        st.error("Please import any :red['**.PDF**'] file to start")

    if st.button("7) Convert to :red[CSV]", disabled=not uploaded_file):
        st.toast(':orange[Converting...]')
        csv_data = convert_to_csv(uploaded_file)
        st.success("CSV created :green[successfully]")
        st.download_button(
            label="Download :red[CSV]",
            data=csv_data,
            file_name="output.csv",
            mime="text/csv"
        )


def convert_to_csv(uploaded_file):
    output = StringIO()
    writer = csv.writer(output)

    reader = PyPDF2.PdfFileReader(uploaded_file)
    for page_num in range(reader.numPages):
        page = reader.getPage(page_num)
        text = page.extractText().split('\n')
        for line in text:
            writer.writerow([line])

    return output.getvalue()


if __name__ == "__main__":
    main()
