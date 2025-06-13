import io

import pandas as pd
import streamlit as st


def main():
    st.title(":orange[Text] to :blue[XLSX] Converter")

    option = st.radio("7) Select input method:", ("Enter Text", "Upload Text File"))

    if option == "Enter Text":
        input_text = st.text_area("7) Enter your text here:")
    else:
        uploaded_file = st.file_uploader("7) Upload Text File", type=["txt"])
        if uploaded_file is not None:
            input_text = uploaded_file.getvalue().decode("utf-8")
            st.success('Data read :green[**Successfully**]')
        else:
            st.error("Please import any :red['**.TXT**'] file to start")

    if st.button("Convert to :blue[XLSX]"):
        if 'input_text' in locals():
            xlsx_file = convert_to_xlsx(input_text)
            st.toast(':orange[Converting...]')
            st.success("XLSX created :green[successfully]")
            st.download_button("Download :blue[XLSX]", xlsx_file, file_name="output.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        else:
            st.warning("Please provide some text to convert.")


def convert_to_xlsx(text):
    rows = text.strip().split('\n')

    df = pd.DataFrame(rows, columns=['Text'])
    xlsx_output = io.BytesIO()

    df.to_excel(xlsx_output, index=False)
    xlsx_output.seek(0)
    return xlsx_output.getvalue()


if __name__ == "__main__":
    main()
