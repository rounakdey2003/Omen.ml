import streamlit as st

from omen_python import (text_pdf, text_html, text_docx, text_doc, text_epub, text_xml, text_xlsx, text_csv,
                         pdf_text, pdf_html, pdf_docx, pdf_doc, pdf_xml, pdf_epub, pdf_csv, pdf_xlsx)

with st.spinner('Loading...'):
    st.toast('Checking Environment')


    def main():
        titleCol1, titleCol2, titleCol3 = st.columns([1, 1, 1])
        with titleCol1:
            pass
        with titleCol2:
            st.title('***ToolKit*** 🛠️')
            st.write(':grey[Everyday converters for PDF, Text, Docx...]')
        with titleCol3:
            pass

        st.divider()

        st.header("***Text*** :grey[to Everything]")
        st.write("##")
        text_col1, text_col2 = st.columns([1, 1])

        with text_col1:
            with st.expander("Text to :red[PDF]"):
                text_pdf.main()

            with st.expander("Text to :blue[DOCX]"):
                text_docx.main()

            with st.expander("Text to :red[HTML]"):
                text_html.main()

            with st.expander("Text to :blue[XLSX]"):
                text_xlsx.main()

        with text_col2:
            with st.expander("Text to :blue[EPUB]"):
                text_epub.main()

            with st.expander("Text to :red[DOC]"):
                text_doc.main()

            with st.expander("Text to :blue[XML]"):
                text_xml.main()

            with st.expander("Text to :red[CSV]"):
                text_csv.main()

        st.divider()

        st.header("***PDF*** :grey[to Everything]")
        st.write("##")
        pdf_col1, pdf_col2 = st.columns([1, 1])

        with pdf_col1:
            with st.expander("PDF to :red[Text]"):
                pdf_text.main()

            with st.expander("PDF to :blue[DOCX]"):
                pdf_docx.main()

            with st.expander("PDF to :red[HTML]"):
                pdf_html.main()

            with st.expander("PDF to :blue[XLSX]"):
                pdf_xlsx.main()

        with pdf_col2:
            with st.expander("PDF to :blue[EPUB]"):
                pdf_epub.main()

            with st.expander("PDF to :red[DOC]"):
                pdf_doc.main()

            with st.expander("PDF to :blue[XML]"):
                pdf_xml.main()

            with st.expander("PDF to :red[CSV]"):
                pdf_csv.main()


    if __name__ == "__main__":
        main()

    st.toast(':green[Ready!]')
