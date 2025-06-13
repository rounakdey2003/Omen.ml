from io import BytesIO

import PyPDF2
import streamlit as st
from docx import Document


def main():
    st.title(":orange[PDF] to :blue[DOCX] Converter")

    uploaded_file = st.file_uploader("3) Upload PDF File", type=["pdf"])
    if uploaded_file is not None:
        st.success('Data read :green[**Successfully**]')
    else:
        st.error("Please import any :red['**.PDF**'] file to start")

    if st.button("3) Convert to :blue[DOCX]", disabled=not uploaded_file):
        st.toast(':orange[Converting...]')
        docx_bytes = convert_to_docx(uploaded_file)
        st.success("DOCX created :green[successfully]")
        st.download_button(
            label="Download :blue[DOCX]",
            data=docx_bytes,
            file_name="output.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )


def convert_to_docx(uploaded_file):
    doc = Document()

    reader = PyPDF2.PdfFileReader(uploaded_file)
    for page_num in range(reader.numPages):
        page = reader.getPage(page_num)
        text = page.extractText()
        doc.add_paragraph(text)

    docx_bytes = BytesIO()
    doc.save(docx_bytes)
    return docx_bytes.getvalue()


if __name__ == "__main__":
    main()
