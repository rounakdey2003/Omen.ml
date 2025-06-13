import streamlit as st
from fpdf import FPDF


def main():
    st.title(":orange[Text] to :red[PDF] Converter")    

    option = st.radio("Select input method:", ("Enter Text", "Upload Text File"))

    if option == "Enter Text":
        input_text = st.text_area("Enter your text here:")
    else:
        uploaded_file = st.file_uploader("Upload Text File", type=["txt"])
        if uploaded_file is not None:

            input_text = uploaded_file.getvalue().decode("utf-8")

            st.success('Data read :green[**Successfully**]')
        else:
            st.error("Please import any :red['**.TXT**'] file to start")

    if st.button("Convert to :red[PDF]"):
        if 'input_text' in locals():
            st.toast(':orange[Converting...]')
            pdf = convert_to_pdf(input_text)
            st.success("PDF created :green[successfully]")
            st.download_button("Download :red[PDF]", pdf.output(dest="S").encode("latin1"), file_name="output.pdf",
                               mime="application/pdf")
        else:
            st.warning("Please provide some text to convert.")


def convert_to_pdf(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt=text)
    return pdf


if __name__ == "__main__":
    main()
