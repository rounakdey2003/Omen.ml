import PyPDF2
import streamlit as st


def main():
    st.title(":orange[PDF] to :red[Text] Converter")   

    uploaded_file = st.file_uploader("Upload PDF File", type=["pdf"])
    if uploaded_file is not None:

        st.success('Data read :green[**Successfully**]')
    else:
        st.error("Please import any :red['**.PDF**'] file to start")

    if st.button("Convert to :red[Text]", disabled=not uploaded_file):
        st.toast(':orange[Converting...]')
        input_text = convert_to_text(uploaded_file)
        st.success("Text created :green[successfully]")
        st.download_button(
            label="Download :red[Text]",
            data=input_text,
            file_name="output.txt",
            mime="text/plain"
        )


def convert_to_text(uploaded_file):
    text = ""

    reader = PyPDF2.PdfFileReader(uploaded_file)
    for page_num in range(reader.numPages):
        page = reader.getPage(page_num)
        text += page.extractText()
    return text


if __name__ == "__main__":
    main()
