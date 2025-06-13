import xml.etree.ElementTree

import streamlit as st


def main():
    st.title(":orange[Text] to :blue[XML] Converter")

    option = st.radio("6) Select input method:", ("Enter Text", "Upload Text File"))

    if option == "Enter Text":
        input_text = st.text_area("6) Enter your text here:")
    else:
        uploaded_file = st.file_uploader("6) Upload Text File", type=["txt"])
        if uploaded_file is not None:
            input_text = uploaded_file.getvalue().decode("utf-8")
            st.success('Data read :green[**Successfully**]')
        else:
            st.error("Please import any :red['**.TXT**'] file to start")

    if st.button("Convert to :blue[XML]"):
        if 'input_text' in locals():
            xml_content = convert_to_xml(input_text)
            st.toast(':orange[Converting...]')
            st.success("XML created :green[successfully]")
            st.download_button("Download :blue[XML]", xml_content, file_name="output.xml", mime="text/xml")
        else:
            st.warning("Please provide some text to convert.")


def convert_to_xml(text):
    root = xml.etree.ElementTree.Element("root")
    text_node = xml.etree.ElementTree.SubElement(root, "text")
    text_node.text = text
    xml_str = xml.etree.ElementTree.tostring(root, encoding='unicode', method='xml')
    return xml_str


if __name__ == "__main__":
    main()
