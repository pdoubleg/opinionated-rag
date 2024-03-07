import base64

import pdfkit
import streamlit as st


def html2pdf(url, pdf_file):
    pdfkit.from_url(url, pdf_file)


def readpdf(pdf_file):
    with open(pdf_file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf">'
    return pdf_display

def show_pdf(file_path):
    with open(file_path,"rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="800" height="800" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

if __name__ == "__main__":
    st.set_page_config(page_title="HTML2PDF", page_icon='✅',
        initial_sidebar_state='collapsed')
    st.title('🔨 HTML2PDF Test for Streamlit Sharing')
    st.markdown("""
        This app is only a very simple test for html2pdf on **Streamlit Sharing** runtime. <br>
        The suggestion for this demo app came from a post on the Streamlit Community Forum.  <br>
        <https://discuss.streamlit.io/t/deploying-an-app-using-using-wkhtmltopdf-on-herokou/12029>  <br>

        This is just a very very simple example and a proof of concept.
        A link is called and converted HTML to PDF.
        Afterwards the pdf file is displayed.

        The downside is that websites with content created by javascript will probably not be rendered properly.
        Onyl static website content will be downloaded and therefore rendered to pdf.
        """, unsafe_allow_html=True)
    st.balloons()
    if st.button('Start html2pdf run'):
        pdf_file = "outputs/pseudo_shep.pdf"
        show_pdf(pdf_file)

        