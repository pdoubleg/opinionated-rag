import asyncio
import imghdr
import logging
import os
import sys
from datetime import datetime
from bs4 import BeautifulSoup
import openai
import pandas as pd
from contextlib import contextmanager
from typing import Any, Iterator, Optional
from rich import print as rprint
from rich.text import Text
import aiofiles
import urllib
import uuid
from pathlib import Path
from md2pdf.core import md2pdf
import jinja2
import pdfkit

import imghdr
from io import BytesIO
import concurrent.futures
import shutil
import tempfile
import pypdfium2 as pdfium
import pytesseract
from pytesseract import image_to_string
from PIL import Image

from src.utils.llm import prompt_multi_image_input
from src.utils.configuration import settings


from datetime import datetime
import functools
import inspect
import tempfile
import markdown

@functools.lru_cache(maxsize=1000)
def download_temp_file(file_id: str, suffix: str = None):
    """
    Downloads a file from OpenAI's servers and saves it to a temporary file.

    Args:
        file_id: The ID of the file to be downloaded.
        suffix: The file extension to be used for the temporary file.

    Returns:
        The file path of the downloaded temporary file.
    """

    client = openai.OpenAI()
    response = client.files.content(file_id)

    # Create a temporary file with a context manager to ensure it's cleaned up
    # properly
    temp_file = tempfile.NamedTemporaryFile(delete=False, mode="wb", suffix=suffix)
    temp_file.write(response.content)

    return temp_file.name


def format_message_string(message):
    timestamp = (
        datetime.fromtimestamp(message.created_at).strftime("%H:%M:%S %p").lstrip("0")
    )
    content = []
    for item in message.content:
        if item.type == "text":
            content.append(item.text.value + "\n\n")
        elif item.type == "image_file":
            # Use the download_temp_file function to download the file and get
            # the local path
            local_file_path = download_temp_file(item.image_file.file_id, suffix=".png")
            content.append(
                f"*View attached image: [{local_file_path}]({local_file_path})*"
            )

    for attachment in message.attachments:
        content.append(f"Attached file: {attachment}\n")

    out_str = f"**{str(message.role.title())}**\n\n"
    out_str = str(timestamp)
    out_str = inspect.cleandoc("\n\n".join(content))

    return out_str


async def write_to_file(filename: str, text: str) -> None:
    """
    Asynchronously write text to a file in UTF-8 encoding.

    Args:
        filename (str): The filename to write to.
        text (str): The text to write.
    """
    # Convert text to UTF-8, replacing any problematic characters
    text_utf8 = text.encode("utf-8", errors="replace").decode("utf-8")

    async with aiofiles.open(filename, "w", encoding="utf-8") as file:
        await file.write(text_utf8)


async def write_md_to_pdf(
    text: str, file_name: Optional[str] = None, output_dir: Optional[Path] = None
) -> Optional[str]:
    """
    Converts Markdown text to a PDF file and returns the file path. Allows specifying an output directory.

    Args:
        text (str): Markdown text to convert.
        file_name (Optional[str]): Optional custom file name for the output PDF.
        output_dir (Optional[Path]): Optional directory to write the PDF file to. Defaults to './outputs'.

    Returns:
        Optional[str]: The encoded file path of the generated PDF or None if an error occurs.
    """
    if not file_name:
        file_name = datetime.now().strftime("%Y%m%d") + "-" + uuid.uuid4().hex[:5]
    if not output_dir:
        output_dir = "./outputs"
    file_path = f"{output_dir}/{file_name}"
    directory = os.path.dirname(file_path)
    # Ensure the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)

    await write_to_file(f"{file_path}.md", text)

    try:
        await asyncio.to_thread(
            md2pdf,
            f"{file_path}.pdf",
            md_content=None,
            md_file_path=f"{file_path}.md",
            css_file_path="./src/style.css",
            base_url=None,
        )
        print(f"Report written to {file_path}.pdf")
    except Exception as e:
        print(f"Error in converting Markdown to PDF: {e}")
        return None
    encoded_file_path = urllib.parse.quote(f"{file_path}.pdf")
    return encoded_file_path


def stringify(x: Any) -> str:
    # Convert x to DataFrame if it is not one already
    if isinstance(x, pd.Series):
        df = x.to_frame()
    elif not isinstance(x, pd.DataFrame):
        return str(x)
    else:
        df = x

    # Truncate long text columns to 1000 characters
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(
                lambda item: (item[:1000] + "...")
                if isinstance(item, str) and len(item) > 1000
                else item
            )

    # Limit to 10 rows
    df = df.head(10)

    # Convert to string
    return df.to_string(index=False)  # type: ignore


def shorten_text(text: str, chars: int = 40) -> str:
    text = " ".join(text.split())
    return text[:chars] + "..." + text[-chars:] if len(text) > 2 * chars else text
     
        

def generate_html_with_toc(html_text: str) -> str:
    """
    Generates HTML content with a table of contents (ToC) based on the headers found in the input HTML text.
    
    Args:
        html_text (str): The HTML content converted from Markdown.
        
    Returns:
        str: The HTML content with a ToC inserted at the beginning.
    """
    soup = BeautifulSoup(html_text, 'html.parser')

    toc_list = soup.new_tag("ul", id='table-of-contents')
    back_to_toc_template = soup.new_tag("a", href="#table-of-contents")
    back_to_toc_template.string = "Back to Table of Contents"
    back_to_toc_template['class'] = 'back-to-toc'

    for header in soup.find_all(['h1', 'h2', 'h3']):
        header_id = header.text.replace(" ", "-").lower()
        header['id'] = header_id  # Assign ID to header for linking
        
        # Create ToC entry
        li = soup.new_tag("li")
        a = soup.new_tag("a", href=f"#{header_id}")
        a.string = header.text
        li.append(a)
        toc_list.append(li)
        
        # Add back-link after the header or section
        back_link = soup.new_tag("a", href="#table-of-contents", **{'class': 'back-to-toc'})
        back_link.string = "Back to Table of Contents"
        header.insert_after(back_link)

    # Check if the body tag exists and insert the ToC, otherwise prepend to the soup object
    if soup.body:
        soup.body.insert(0, toc_list)
    else:
        soup.insert(0, toc_list)

    return str(soup)



prompt_template = """
The following document contains anonymized student report cards that have been processed with OCR and a language model to improve overall usability. \
Each student, indicated by ID number, is eligible for a variety of academic scholarships designed for different interests and skill sets. \
To help introduce each student to the review committee who will be evaluating the details, please draft an executive summary style overview. Include quantitative observations where possible. \
Be concise, clear and use a scholarly tone. The committee has many candidates to review so it is important that the summary be information dense while maintaining readability. \
You will received a student ID number and the document. Please focus only on the given ID using information found in the 'report card' text and 'Attendance Count Report'. \

Student ID number: **{student_id}**
Here is the document: {pdf_result}
"""


report_card_prompt = f"""
You are a world class document processing AI who helps students convert their paper report cards to a more structured format so they can better \
understand and monitor their performance. Using the image, generate a well-structured and nicely formatted report. \
It should contain four sections: \
 * Course History \
 * GPA and Credits \
 * Work In Progress \
 * Summary \
   - Strengths \
   - Opportunities \

ALWAYS begin with the **Student ID** number and end with a concise summary of strengths and opportunities. \
Here is a list of valid Student IDs that we expect to find: '583619', '637408', '670038', '714551', '592686', '660835', '604327', '610070', '635083', '578140'. \
**NOTE:** If the image is an Attendance Report respond with "None".
"""

template_str = '''
<table class="poll-table">
  <thead>
    <tr>
      {% for c in columns %}
      <th>{{ c.title() }}</th>
      {% endfor %}
    </tr>
  </thead>
  <tbody>
     {% for row in rows %}
     <tr>
     {% for k, v in row.items() %}
      <td>
        {% if k == 'title' %}
        <!-- Link the title to the item's URL -->
        <a href="{{ row['listing_url'] }}" style="color: #000; text-decoration: none;">{{ v }}</a>
        {% elif k == 'listing_url' %}
        <!-- Skip rendering listing_url as it's already linked with title -->
        {% else %}
        {{ v }}
        {% endif %}
     </td>
     {% endfor %}
     </tr>
     {% endfor %}
  </tbody>
</table>

<style>
.poll-table {
  max-width: 960px; 
  width: 100%;
  border-collapse: collapse;
  font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
  margin: 0 auto;
  box-shadow: 0 2px 3px rgba(0,0,0,0.1);
}

.poll-table th,
.poll-table td {
  page-break-inside: avoid; 
  border: 1px solid #ddd;
  padding: 10px 15px;
  text-align: left;
  font-size: 14px;
  border-left: none;
  border-right: none;
  overflow: hidden;
  text-overflow: ellipsis;
}

.poll-table th {
  background-color: #000;
  color: #fff;
  font-weight: bold;
  font-size: 24px;
  overflow: hidden;
}

.poll-table tr:nth-child(even) {
  background-color: #f9f9f9;
}

.poll-table tr:hover {
  background-color: #f1f1f1;
}

.poll-table tbody td {
  border-bottom: 1px solid #ddd;
}

.poll-table  {
  border: 1px solid #000;
}

.poll-table td:first-child,
.poll-table th:first-child {
  border-left: 1px solid #ddd;
}

.poll-table td:last-child,
.poll-table th:last-child {
  border-right: 1px solid #ddd;
}
</style>
'''


def render_page(pdf_file, page_index, scale):
    renderer = pdf_file.render(
        pdfium.PdfBitmap.to_pil,
        page_indices=[page_index],
        scale=scale,
    )
    image_list = list(renderer)
    image = image_list[0]
    image_byte_array = BytesIO()
    image.save(image_byte_array, format='jpeg', optimize=True)
    image_byte_array = image_byte_array.getvalue()
    return {page_index: image_byte_array}


def convert_pdf_to_images(file_path, scale=300/72):
    # Check if the file is already an image
    if imghdr.what(file_path) is not None:
        # If it is, return it as is
        with open(file_path, 'rb') as f:
            return [{0: f.read()}]

    # If it's not an image, proceed with the conversion
    pdf_file = pdfium.PdfDocument(file_path)
    
    page_indices = [i for i in range(len(pdf_file))]
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for i in page_indices:
            future = executor.submit(render_page, pdf_file, i, scale)
            futures.append(future)
        
        final_images = []
        for future in concurrent.futures.as_completed(futures):
            final_images.append(future.result())
    
    return final_images


def process_image(index, image_bytes):
    try:
        image = Image.open(BytesIO(image_bytes))
        raw_text = str(image_to_string(image))
        return raw_text
    except Exception as e:
        raise Exception(f"Error processing image {index}: {e}")


def extract_text_with_pytesseract(list_dict_final_images):
    
    image_list = [list(data.values())[0] for data in list_dict_final_images]
    image_content = []
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for index, image_bytes in enumerate(image_list):
            future = executor.submit(process_image, index, image_bytes)
            futures.append(future)
        
        for future in concurrent.futures.as_completed(futures):
            try:
                raw_text = future.result()
                image_content.append(raw_text)
            except Exception as e:
                raise Exception(f"Error processing image: {e}")
    
    return image_content


def process_file(file_path: str) -> str:
    """
    Process a file at the given path and extract text from it.
    
    Args:
        file_path (str): The path to the file to process.
        
    Returns:
        str: The extracted text from the file.
    """
    # Check the file type
    file_type = imghdr.what(file_path)
    if file_type is None:
        # If the file is not an image, assume it's a PDF and extract the text from it
        images = convert_pdf_to_images(file_path)
        extracted_text = extract_text_with_pytesseract(images)
        input_data = "\n\n new page --- \n\n".join(extracted_text)
    else:
        # If the file is an image or text, read it directly
        with open(file_path, 'r') as f:
            input_data = f.read()
    
    return input_data


def extract_text_with_pytesseract_and_gpt4(list_dict_final_images, prompt):
    image_list = [list(data.values())[0] for data in list_dict_final_images]
    image_content = []
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for index, image_bytes in enumerate(image_list):
            future = executor.submit(process_image_with_gpt4, index, image_bytes, prompt)
            futures.append(future)
        
        for future in concurrent.futures.as_completed(futures):
            try:
                raw_text = future.result()
                image_content.append(raw_text)
            except Exception as e:
                raise Exception(f"Error processing image: {e}")
    
    return image_content


def process_image_with_gpt4(index, image_bytes, prompt):
    try:
        # Save the image bytes to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            temp_file.write(image_bytes)
            temp_file_path = temp_file.name
        
        # Process the image using GPT-4 vision model
        gpt4_response = prompt_multi_image_input(prompt=prompt, image_paths=[temp_file_path], max_tokens=3000)
        
        # Extract text using pytesseract
        image = Image.open(BytesIO(image_bytes))
        ocr_text = str(image_to_string(image))
        
        # Combine the GPT-4 response and OCR text
        combined_text = f"# GPT-4-vision Summary:\n\n{gpt4_response}\n\n# OCR Text:\n\n{ocr_text}\n\n"
        
        return combined_text
    except Exception as e:
        raise Exception(f"Error processing image {index}: {e}")
    finally:
        # Remove the temporary file
        os.unlink(temp_file_path)


def process_file_with_gpt4(file_path: str, prompt: str) -> str:
    """
    Process a file at the given path and extract text from it using GPT-4 vision and pytesseract.
    
    Args:
        file_path (str): The path to the file to process.
        prompt (str): The prompt to use for GPT-4 vision processing.
        
    Returns:
        str: The extracted text from the file.
    """
    # Check the file type
    file_type = imghdr.what(file_path)
    if file_type is None:
        # If the file is not an image, assume it's a PDF and extract the text from it
        images = convert_pdf_to_images(file_path)
        extracted_text = extract_text_with_pytesseract_and_gpt4(images, prompt)
        input_data = f"\n\n {'-'*50} New Page {'-'*50} \n\n".join(extracted_text)
    else:
        # If the file is an image or text, read it directly
        with open(file_path, 'r') as f:
            input_data = f.read()
    
    return input_data

async def convert_html_to_pdf(
    html_string: str,
    image_url: str,
    template_folder: str = "./static",
    base_html_file: str = "base.html",
    output_file: str = "generate_pdf.pdf",
    output_folder: str = "./data",
) -> str:

    if output_folder.endswith("/"):
        raise ValueError("Wrong output folder name, should not end with '/'")
    else:
        pdf_file_name = f"{output_folder}/{output_file}"

    try:
        template_loader = jinja2.FileSystemLoader(template_folder)
        template_env = jinja2.Environment(loader=template_loader)

        basic_template = template_env.get_template(base_html_file)

        output_html_code = basic_template.render()
        # print(output_html_code)

        # render content, this if for once we have AI generated response
        output_html_code = basic_template.render(
            ai_generated_content=html_string,
            image_url=image_url,
        )

        options = {
            'page-size': 'A4',
            'margin-top': '0.75in',
            'margin-bottom': '0.75in',
            'margin-right': '0.55in',
            'margin-left': '0.55in',
            'encoding': "UTF-8",
            'footer-right': '[page] of [topage]',
            'footer-font-size': "9",
            'custom-header': [
                ('Accept-Encoding', 'gzip')
            ],
            'enable-local-file-access': False,
            'no-outline': None,
            'enable-local-file-access': False,
            'no-outline': None
        }

        pdfkit.from_string(
            input=output_html_code,
            output_path=pdf_file_name,
            options=options
        )

    except Exception as e:
        print(e)
        return ""

    return pdf_file_name


# Example usage
# import asyncio
# import markdown
# import nest_asyncio
# nest_asyncio.apply()

# html_str = markdown.markdown(output_string, extensions=['tables', 'nl2br', 'md_in_html'])

# generated_pdf_file_name = asyncio.run(
#     convert_html_to_pdf(
#     html_string=html_str,
#     image_url="",
# ))