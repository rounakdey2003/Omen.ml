from io import BytesIO

import PyPDF2
import pandas as pd
import streamlit as st


def main():
    st.title(":orange[PDF] to :blue[XLSX] Converter")   

    uploaded_file = st.file_uploader("8) Upload PDF File", type=["pdf"])
    if uploaded_file is not None:
        st.success('Data read :green[**Successfully**]')
    else:
        st.error("Please import any :red['**.PDF**'] file to start")

    if st.button("8) Convert to :blue[XLSX]", disabled=not uploaded_file):
        st.toast(':orange[Converting...]')
        excel_data = convert_to_excel(uploaded_file)
        st.success("Excel file created :green[successfully]")
        st.download_button(
            label="Download :blue[XLSX]",
            data=excel_data,
            file_name="output.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )


def convert_to_excel(uploaded_file):
    text = ""
    reader = PyPDF2.PdfFileReader(uploaded_file)
    for page_num in range(reader.numPages):
        page = reader.getPage(page_num)
        text += page.extractText()

    # Convert text to DataFrame
    data = {'Text': [text]}
    df = pd.DataFrame(data)

    # Create Excel file in memory
    excel_writer = pd.ExcelWriter(BytesIO(), engine='xlsxwriter')
    df.to_excel(excel_writer, index=False)
    excel_writer.save()
    excel_data = excel_writer.bytes.getvalue()

    return excel_data


if __name__ == "__main__":
    main()
