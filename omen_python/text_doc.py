import io

import streamlit as st
from docx import Document


def main():
    st.title(":orange[Text] to :red[DOC] Converter")    

    option = st.radio("4) Select input method:", ("Enter Text", "Upload Text File"))

    if option == "Enter Text":
        input_text = st.text_area("4) Enter your text here:")
    else:
        uploaded_file = st.file_uploader("4) Upload Text File", type=["txt"])
        if uploaded_file is not None:
            input_text = uploaded_file.getvalue().decode("utf-8")
            st.success('Data read :green[**Successfully**]')
        else:
            st.error("Please import any :red['**.TXT**'] file to start")

    if st.button("Convert to :red[DOC]"):
        if 'input_text' in locals():
            docx_file = convert_to_doc(input_text)
            st.toast(':orange[Converting...]')
            st.success("DOC created :green[successfully]")
            st.download_button("Download :red[DOC]", docx_file, file_name="output.doc",
                               mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        else:
            st.warning("Please provide some text to convert.")


def convert_to_doc(text):
    doc = Document()
    doc.add_paragraph(text)
    docx_output = io.BytesIO()
    doc.save(docx_output)
    docx_output.seek(0)
    return docx_output.getvalue()


if __name__ == "__main__":
    main()
