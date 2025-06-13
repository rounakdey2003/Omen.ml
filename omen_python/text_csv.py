import csv
import io

import streamlit as st


def main():
    st.title(":orange[Text] to :red[CSV] Converter")    

    option = st.radio("8) Select input method:", ("Enter Text", "Upload Text File"))

    if option == "Enter Text":
        input_text = st.text_area("8) Enter your text here:")
    else:
        uploaded_file = st.file_uploader("8) Upload Text File", type=["txt"])
        if uploaded_file is not None:
            input_text = uploaded_file.getvalue().decode("utf-8")
            st.success('Data read :green[**Successfully**]')
        else:
            st.error("Please import any :red['**.TXT**'] file to start")

    if st.button("Convert to :red[CSV]"):
        if 'input_text' in locals():
            csv_file = convert_to_csv(input_text)
            st.toast(':orange[Converting...]')
            st.success("CSV created :green[successfully]")
            st.download_button("Download :red[CSV]", csv_file, file_name="output.csv", mime="text/csv")
        else:
            st.warning("Please provide some text to convert.")


def convert_to_csv(text):
    # Convert text to CSV format
    csv_output = io.StringIO()
    writer = csv.writer(csv_output)
    writer.writerow(["Text"])
    writer.writerow([text])
    csv_output.seek(0)
    return csv_output.getvalue()


if __name__ == "__main__":
    main()
