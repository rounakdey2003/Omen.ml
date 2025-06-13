import csv
from io import BytesIO

import streamlit as st
from fpdf import FPDF


def main():
    st.title(":orange[CSV] to :blue[PDF] Converter")

    uploaded_file = st.file_uploader("9) Upload CSV File", type=["csv"])
    if uploaded_file is not None:
        st.success('Data read :green[**Successfully**]')
    else:
        st.error("Please import any :red['**.CSV**'] file to start")

    if st.button("9) Convert to :blue[PDF]", disabled=not uploaded_file):
        st.toast(':orange[Converting...]')
        pdf_data = convert_to_pdf(uploaded_file)
        st.success("PDF created :green[successfully]")
        st.download_button(
            label="Download :blue[PDF]",
            data=pdf_data,
            file_name="output.pdf",
            mime="application/pdf"
        )


def convert_to_pdf(uploaded_file):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)

    reader = csv.reader(uploaded_file)
    for row in reader:
        for item in row:
            pdf.cell(200, 10, txt=item, ln=True, align="L")

    pdf_output = BytesIO()
    pdf_output.write(pdf.output(dest='S').encode('latin1'))
    return pdf_output.getvalue()


if __name__ == "__main__":
    main()
