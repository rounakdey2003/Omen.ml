import streamlit as st


def main():
    def convert_to_html(text):
        return f"<div>{text}</div>"

    st.title(":orange[Text] to :red[HTML] Converter")    

    option = st.radio("5) Select input method:", ("Enter Text", "Upload Text File"))

    if option == "Enter Text":
        input_text = st.text_area("5) Enter your text here:")
    else:
        uploaded_file = st.file_uploader("5) Upload Text File", type=["txt"])
        if uploaded_file is not None:

            input_text = uploaded_file.getvalue().decode("utf-8")

            st.success('Data read :green[**Successfully**]')
        else:
            st.error("Please import any :red['**.TXT**'] file to start")

    if st.button("Convert to :red[HTML]"):
        if 'input_text' in locals():
            st.toast(':orange[Converting...]')
            html_content = convert_to_html(input_text)
            st.success("HTML created :green[successfully]")
            st.download_button("Download HTML", html_content, file_name="output.html", mime="text/html")
        else:
            st.warning("Please provide some text to convert.")


if __name__ == "__main__":
    main()
