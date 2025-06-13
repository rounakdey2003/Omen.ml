import PyPDF2
import streamlit as st


def main():
    st.title(":orange[PDF] to :blue[XML] Converter")    

    uploaded_file = st.file_uploader("5) Upload PDF File", type=["pdf"])
    if uploaded_file is not None:
        st.success('Data read :green[**Successfully**]')
    else:
        st.error("Please import any :red['**.PDF**'] file to start")

    if st.button("5) Convert to :blue[XML]", disabled=not uploaded_file):
        st.toast(':orange[Converting...]')
        text_output = convert_to_text(uploaded_file)
        xml_output = convert_to_xml(text_output)
        st.success("XML created :green[successfully]")
        st.download_button(
            label="Download :blue[XML]",
            data=xml_output,
            file_name="output.xml",
            mime="text/xml",
            key="download-xml"
        )


def convert_to_text(uploaded_file):
    text = ""

    reader = PyPDF2.PdfFileReader(uploaded_file)
    for page_num in range(reader.numPages):
        page = reader.getPage(page_num)
        text += page.extractText()
    return text


def convert_to_xml(text):
    xml_data = f"<text>\n{text}\n</text>"
    return xml_data


if __name__ == "__main__":
    main()
