import re

import PyPDF2
import streamlit as st


def main():
    st.title(":orange[PDF] to :red[HTML] Converter")

    uploaded_file = st.file_uploader("2) Upload PDF File", type=["pdf"])
    if uploaded_file is not None:
        st.success('Data read :green[**Successfully**]')
    else:
        st.error("Please import any :red['**.PDF**'] file to start")

    if st.button("2) Convert to :red[HTML]", disabled=not uploaded_file):
        st.toast(':orange[Converting...]')
        text_output = convert_to_text(uploaded_file)
        html_output = convert_to_html(text_output)
        st.success("HTML created :green[successfully]")
        st.download_button(
            label="Download :red[HTML]",
            data=html_output,
            file_name="output.html",
            mime="text/html"
        )


def convert_to_text(uploaded_file):
    text = ""

    reader = PyPDF2.PdfFileReader(uploaded_file)
    for page_num in range(reader.numPages):
        page = reader.getPage(page_num)
        text += page.extractText()
    return text


def convert_to_html(text):
    html_output = re.sub(r'\n+', '<br>', text)

    html_output = f"<p>{html_output}</p>"

    return html_output


if __name__ == "__main__":
    main()
