import io

import PyPDF2
import streamlit as st
from docx import Document


def main():
    st.title(":orange[PDF] to :blue[DOC] Converter")

    uploaded_file = st.file_uploader("4) Upload PDF File", type=["pdf"])
    if uploaded_file is not None:
        st.success('Data read :green[**Successfully**]')
    else:
        st.error("Please import any :red['**.PDF**'] file to start")

    if st.button("4) Convert to :blue[DOC]", disabled=not uploaded_file):
        st.toast(':orange[Converting...]')
        docx_file = convert_to_doc(uploaded_file)
        st.success("DOC created :blue[successfully]")
        st.download_button(
            label="Download :blue[DOC]",
            data=docx_file,
            file_name="output.doc",
            mime="application/octet-stream"
        )


def convert_to_doc(uploaded_file):
    document = Document()
    reader = PyPDF2.PdfFileReader(uploaded_file)
    for page_num in range(reader.numPages):
        page = reader.getPage(page_num)
        text = page.extractText()
        document.add_paragraph(text)

    docx_bytes = io.BytesIO()
    document.save(docx_bytes)
    docx_bytes.seek(0)
    return docx_bytes.read()


if __name__ == "__main__":
    main()
