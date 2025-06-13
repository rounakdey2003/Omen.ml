import io

import streamlit as st
from ebooklib import epub


def main():
    st.title(":orange[Text] to :blue[EPUB] Converter")    

    option = st.radio("2) Select input method:", ("Enter Text", "Upload Text File"))

    if option == "Enter Text":
        input_text = st.text_area("2) Enter your text here:")
    else:
        uploaded_file = st.file_uploader("2) Upload Text File", type=["txt"])
        if uploaded_file is not None:
            input_text = uploaded_file.getvalue().decode("utf-8")
            st.success('Data read :green[**Successfully**]')
        else:
            st.error("Please import any :red['**.TXT**'] file to start")

    if st.button("Convert to :blue[EPUB]"):
        if 'input_text' in locals():
            epub_file = convert_to_epub(input_text)
            st.toast(':orange[Converting...]')
            st.success("EPUB created :green[successfully]")
            st.download_button("Download :blue[EPUB]", epub_file, file_name="output.epub", mime="application/epub+zip")
        else:
            st.warning("Please provide some text to convert.")


def convert_to_epub(text):
    book = epub.EpubBook()
    book.set_identifier('text_to_epub')
    book.set_title('Converted EPUB')
    book.set_language('en')

    chapter = epub.EpubHtml(title='Chapter 1', file_name='chap_01.xhtml', lang='en')
    chapter.content = '<h1>Chapter 1</h1><p>' + text + '</p>'
    book.add_item(chapter)

    book.toc = (epub.Link('chap_01.xhtml', 'Introduction', 'intro'),)

    style = 'BODY {color: white;}'
    nav_css = epub.EpubItem(uid="style_nav", file_name="style/nav.css", media_type="text/css", content=style)
    book.add_item(nav_css)

    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())

    epub_output = io.BytesIO()
    epub.write_epub(epub_output, book, {})
    epub_output.seek(0)
    return epub_output.getvalue()


if __name__ == "__main__":
    main()
