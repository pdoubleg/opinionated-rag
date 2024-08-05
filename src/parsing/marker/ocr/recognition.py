import base64
import imghdr
import tempfile
from itertools import repeat
from typing import List, Optional, Dict

import pypdfium2 as pdfium
import io
from PIL import Image
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

from surya.ocr import run_recognition
from pytesseract import image_to_string

from src.parsing.marker.models import setup_recognition_model
from src.parsing.marker.ocr.heuristics import should_ocr_page, no_text_found, detect_bad_ocr
from src.parsing.marker.ocr.lang import langs_to_ids
from src.parsing.marker.pdf.images import render_image
from src.parsing.marker.schema.page import Page
from src.parsing.marker.schema.block import Block, Line, Span
from src.parsing.marker.settings import settings
from src.parsing.marker.pdf.extract_text import get_text_blocks


def get_batch_size():
    if settings.RECOGNITION_BATCH_SIZE is not None:
        return settings.RECOGNITION_BATCH_SIZE
    elif settings.TORCH_DEVICE_MODEL == "cuda":
        return 32
    elif settings.TORCH_DEVICE_MODEL == "mps":
        return 32
    return 32


def run_ocr(doc, pages: List[Page], langs: List[str], rec_model, batch_multiplier=1) -> (List[Page], Dict):
    ocr_pages = 0
    ocr_success = 0
    ocr_failed = 0
    no_text = no_text_found(pages)
    ocr_idxs = []
    for pnum, page in enumerate(pages):
        ocr_needed = should_ocr_page(page, no_text)
        if ocr_needed:
            ocr_idxs.append(pnum)
            ocr_pages += 1

    # No pages need OCR
    if ocr_pages == 0:
        return pages, {"ocr_pages": 0, "ocr_failed": 0, "ocr_success": 0, "ocr_engine": "none"}

    ocr_method = settings.OCR_ENGINE
    if ocr_method is None:
        return pages, {"ocr_pages": 0, "ocr_failed": 0, "ocr_success": 0, "ocr_engine": "none"}
    elif ocr_method == "surya":
        new_pages = surya_recognition(doc, ocr_idxs, langs, rec_model, pages, batch_multiplier=batch_multiplier)
    elif ocr_method == "ocrmypdf":
        new_pages = tesseract_recognition(doc, ocr_idxs, langs)
    else:
        raise ValueError(f"Unknown OCR method {ocr_method}")

    for orig_idx, page in zip(ocr_idxs, new_pages):
        if detect_bad_ocr(page.prelim_text) or len(page.prelim_text) == 0:
            ocr_failed += 1
        else:
            ocr_success += 1
            pages[orig_idx] = page

    return pages, {"ocr_pages": ocr_pages, "ocr_failed": ocr_failed, "ocr_success": ocr_success, "ocr_engine": ocr_method}


def surya_recognition(doc, page_idxs, langs: List[str], rec_model, pages: List[Page], batch_multiplier=1) -> List[Optional[Page]]:
    images = [render_image(doc[pnum], dpi=settings.SURYA_OCR_DPI) for pnum in page_idxs]
    processor = rec_model.processor
    selected_pages = [p for i, p in enumerate(pages) if i in page_idxs]

    surya_langs = [langs] * len(page_idxs)
    detection_results = [p.text_lines.bboxes for p in selected_pages]
    polygons = [[b.polygon for b in bboxes] for bboxes in detection_results]

    results = run_recognition(images, surya_langs, rec_model, processor, polygons=polygons, batch_size=int(get_batch_size() * batch_multiplier))

    new_pages = []
    for (page_idx, result, old_page) in zip(page_idxs, results, selected_pages):
        text_lines = old_page.text_lines
        ocr_results = result.text_lines
        blocks = []
        for i, line in enumerate(ocr_results):
            block = Block(
                bbox=line.bbox,
                pnum=page_idx,
                lines=[Line(
                    bbox=line.bbox,
                    spans=[Span(
                        text=line.text,
                        bbox=line.bbox,
                        span_id=f"{page_idx}_{i}",
                        font="",
                        font_weight=0,
                        font_size=0,
                    )
                    ]
                )]
            )
            blocks.append(block)
        page = Page(
            blocks=blocks,
            pnum=page_idx,
            bbox=result.image_bbox,
            rotation=0,
            text_lines=text_lines,
            ocr_method="surya"
        )
        new_pages.append(page)
    return new_pages


def tesseract_recognition(doc, page_idxs, langs: List[str]) -> List[Optional[Page]]:
    pdf_pages = generate_single_page_pdfs(doc, page_idxs)
    with ThreadPoolExecutor(max_workers=settings.OCR_PARALLEL_WORKERS) as executor:
        pages = list(executor.map(_tesseract_recognition, pdf_pages, repeat(langs, len(pdf_pages))))
    return pages


def generate_single_page_pdfs(doc, page_idxs) -> List[io.BytesIO]:
    pdf_pages = []
    for page_idx in page_idxs:
        blank_doc = pdfium.PdfDocument.new()
        blank_doc.import_pages(doc, pages=[page_idx])
        assert len(blank_doc) == 1, "Failed to import page"

        in_pdf = io.BytesIO()
        blank_doc.save(in_pdf)
        in_pdf.seek(0)
        pdf_pages.append(in_pdf)
    return pdf_pages


def _tesseract_recognition(in_pdf, langs: List[str]) -> Optional[Page]:
    import ocrmypdf
    import os

    # Create the data directory if it doesn't exist
    os.makedirs("./data", exist_ok=True)
    
    out_pdf_path = "./data/temp_ocr_output.pdf"

    ocrmypdf.ocr(
        in_pdf,
        out_pdf_path,
        language=langs[0],
        output_type="pdf",
        redo_ocr=None,
        force_ocr=True,
        progress_bar=False,
        optimize=False,
        fast_web_view=1e6,
        skip_big=15,  # skip images larger than 15 megapixels
        tesseract_timeout=settings.TESSERACT_TIMEOUT,
        tesseract_non_ocr_timeout=settings.TESSERACT_TIMEOUT,
    )
    
    new_doc = pdfium.PdfDocument(out_pdf_path)
    blocks, _ = get_text_blocks(new_doc, out_pdf_path, max_pages=1)

    page = blocks[0]
    page.ocr_method = "tesseract"
    return page




def render_page(pdf_file: pdfium.PdfDocument, page_index: int, scale: float) -> Dict[int, bytes]:
    """
    Render a single page of a PDF file as a JPEG image.

    This function takes a PDF file, renders a specific page at the given scale,
    and returns the rendered image as bytes in a dictionary.

    Args:
        pdf_file (pdfium.PdfDocument): The PDF document to render from.
        page_index (int): The index of the page to render.
        scale (float): The scale factor for rendering the page.

    Returns:
        Dict[int, bytes]: A dictionary with the page index as the key and the JPEG image bytes as the value.

    Example:
        >>> pdf_doc = pdfium.PdfDocument("example.pdf")
        >>> rendered_page = render_page(pdf_doc, 0, 2.0)
        >>> print(list(rendered_page.keys())[0])  # Should print 0
        0
    """
    renderer = pdf_file.render(
        pdfium.PdfBitmap.to_pil,
        page_indices=[page_index],
        scale=scale,
    )
    image_list = list(renderer)
    image = image_list[0]
    image_byte_array = io.BytesIO()
    image.save(image_byte_array, format='jpeg', optimize=True)
    image_byte_array = image_byte_array.getvalue()
    return {page_index: image_byte_array}


def convert_pdf_to_images(file_path: str, scale: float = 300/72) -> List[Dict[int, bytes]]:
    """
    Convert a PDF file to a list of images, or return the file as-is if it's already an image.

    This function checks if the given file is an image. If it is, it returns the file content
    as a single-element list. If the file is a PDF, it converts each page to an image.

    Args:
        file_path (str): The path to the file (PDF or image) to be processed.
        scale (float): The scale factor for PDF rendering. Defaults to 300/72 (300 DPI).

    Returns:
        List[Dict[int, bytes]]: A list of dictionaries, where each dictionary contains
        a page number as the key and the corresponding image bytes as the value.

    Example:
        >>> images = convert_pdf_to_images('document.pdf')
        >>> print(len(images))  # Number of pages in the PDF
        5
    """
    # Check if the file is already an image
    if imghdr.what(file_path) is not None:
        # If it is, return it as is
        with open(file_path, 'rb') as f:
            return [{0: f.read()}]

    # If it's not an image, proceed with the conversion
    pdf_file = pdfium.PdfDocument(file_path)
    
    page_indices = list(range(len(pdf_file)))
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(render_page, pdf_file, i, scale) for i in page_indices]
        
        final_images = []
        for future in concurrent.futures.as_completed(futures):
            final_images.append(future.result())
    
    return final_images


def images_to_base64(images: List[Dict[int, bytes]]) -> List[Dict[int, str]]:
    """
    Convert a list of image byte dictionaries to base64-encoded string dictionaries.

    Args:
        images (List[Dict[int, bytes]]): A list of dictionaries, where each dictionary
            contains a page number as the key and the corresponding image bytes as the value.

    Returns:
        List[Dict[int, str]]: A list of dictionaries, where each dictionary contains
            a page number as the key and the corresponding base64-encoded image as the value.

    Example:
        >>> byte_images = convert_pdf_to_images('document.pdf')
        >>> base64_images = images_to_base64(byte_images)
        >>> print(list(base64_images[0].values())[0][:20])  # First 20 characters of base64 string
        '/9j/4AAQSkZJRgABAQEA'
    """
    base64_images = []
    for image_dict in images:
        page_num = list(image_dict.keys())[0]
        image_bytes = image_dict[page_num]
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        base64_images.append({page_num: base64_image})
    return base64_images


def convert_pdf_to_base64_images(file_path: str, scale: float = 300/72) -> List[Dict[int, str]]:

    images = convert_pdf_to_images(file_path, scale)
    return images_to_base64(images)


def process_image(index, image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes))
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