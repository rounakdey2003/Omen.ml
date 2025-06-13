import PyPDF2
import streamlit as st
from ebooklib import epub


def main():
    st.title(":orange[PDF] to :blue[EPUB] Converter")

    uploaded_file = st.file_uploader("6) Upload PDF File", type=["pdf"])
    if uploaded_file is not None:
        st.success('Data read :green[**Successfully**]')
    else:
        st.error("Please import any :red['**.PDF**'] file to start")

    if st.button("6) Convert to :blue[EPUB]", disabled=not uploaded_file):
        st.toast(':orange[Converting...]')
        converted_epub_file = convert_to_epub(uploaded_file)
        st.success("EPUB created :green[successfully]")
        st.download_button(
            label="Download :blue[EPUB]",
            data=converted_epub_file,
            file_name="output.epub",
            mime="application/epub+zip"
        )


def convert_to_epub(uploaded_file):
    book = epub.EpubBook()
    book.set_title("Converted EPUB")
    book.set_language("en")

    reader = PyPDF2.PdfFileReader(uploaded_file)
    for page_num in range(reader.numPages):
        page = reader.getPage(page_num)
        text = page.extractText()

        # Add each pages as a chapter
        chapter_title = f"Page {page_num + 1}"
        chapter = epub.EpubHtml(title=chapter_title, file_name=f'page_{page_num + 1}.xhtml', lang='en')
        chapter.content = f"<html><body><p>{text}</p></body></html>"
        book.add_item(chapter)
        book.toc.append((chapter, chapter_title))

    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())

    epub_file_path = "output.epub"
    epub.write_epub(epub_file_path, book, {})

    with open(epub_file_path, 'rb') as file:
        epub_content = file.read()

    return epub_content


if __name__ == "__main__":
    main()
