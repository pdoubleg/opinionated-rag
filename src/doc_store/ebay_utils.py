import io
import os
from io import BytesIO
import base64
from pathlib import Path
from typing import List, Optional
import numpy as np
from pydantic import ConfigDict
import requests
import asyncio
import textwrap
from PIL import Image as PILImage
import matplotlib.pyplot as plt
import tempfile
import markdown
from weasyprint import HTML, CSS
from concurrent.futures import ThreadPoolExecutor, as_completed
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import EmbeddingFunctionRegistry

from .ebay_scraper import eBayProduct

base_abs_path = os.path.dirname(os.path.abspath(__name__))
base_rel_path = os.path.dirname(os.path.realpath(__name__))


def thumbnail(image, scale=3):
    return image.resize(np.array(image.size)//scale)

registry = EmbeddingFunctionRegistry.get_instance()
clip = registry.get("open-clip").create()

class eBayItems(LanceModel):
    model_config = ConfigDict(
        extra='allow'
    )
    id: str
    text: str
    vector: Vector(clip.ndims()) = clip.VectorField()
    image_uri: str = clip.SourceField()
    title: str
    price: str
    image_base64: str
    hash_id: str
    shipping: Optional[str] = None
    location: Optional[str] = None
    condition: Optional[str] = None
    listing_url: Optional[str] = None
    image_url: Optional[str] = None
    sale_end_date: Optional[str] = None

    @property
    def image(self):
        return PILImage.open(self.image_uri)
    
    @property
    def listing_link(self):
        return f"[Listing link]({str(self.listing_url)}"
    
    @property
    def image_link(self):
        return f"[Image link]({str(self.image_url)})"
    
    @property
    def to_data_dict(self):
        data_dict = {
            "id": self.id,
            "title": self.title,
            "price": self.price,
            "shipping": self.shipping,
            "location": self.location,
            "condition": self.condition,
            "text": self.text,
            "listing_url": str(self.listing_url),
            "image_url": str(self.image_url),
            "image_base64": self.image_base64,
            "image_uri": str(self.image_uri),
            "hash_id": self.hash_id,
        }
        return data_dict
    
    def __str__(self):
        display_str = f"**{self.title}**\n\n{self.condition}\n\n"
        price_str = f"{self.price}\n\n{self.shipping}\n\n"
        display_str += price_str
        # Skipping initial characters e.g. "eBay\n"
        wrapped_description = textwrap.fill(self.text[5:-1], width=100)
        description_str = (
            f"\n\nItem description from the seller:\n\n{wrapped_description}\n\n"
        )
        id_str = f"- eBay item number: {self.id}\n"
        url_str = f"- [Listing link]({str(self.listing_url)})\n\n"
        image_str = f"- [Image link]({str(self.image_url)})\n\n"
        display_str += description_str
        display_str += url_str
        display_str += image_str
        display_str += id_str
        display_str += "-" * 100
        return display_str
    
    
def deduplicate_products(products: List[eBayProduct]) -> List[eBayProduct]:
    """
    Deduplicates a list of eBayProduct instances based on the image.

    Args:
        products (List[eBayProduct]): The list of eBayProduct instances to deduplicate.

    Returns:
        List[eBayProduct]: A list of unique eBayProduct instances based on the image.
    """
    seen_images = set()
    unique_products = []

    for product in products:
        image_base64 = product.item.image_base64
        if image_base64 not in seen_images:
            unique_products.append(product)
            seen_images.add(image_base64)

    return unique_products


def deduplicate_search_items(items: List[eBayItems]) -> List[eBayItems]:
    """
    Deduplicates a list of eBayItems instances based on the image.

    Args:
        products (List[eBayItems]): The list of eBayItems instances to deduplicate.

    Returns:
        List[eBayItems]: A list of unique eBayItems instances based on the image.
    """
    seen_images = set()
    unique_items = []

    for item in items:
        image_base64 = item.image_base64
        if image_base64 not in seen_images:
            unique_items.append(item)
            seen_images.add(image_base64)

    return unique_items


def process_ebay_item_image(item: eBayProduct) -> None:
    """
    Processes an eBay item by fetching and encoding its image, then downloading the image.

    Args:
        item (eBayProduct): The eBay item to process.
    """
    item.item.fetch_and_encode_image()
    item.download_image()

def process_ebay_images_with_threadpool(search_results: List[eBayProduct]) -> None:
    """
    Processes images for all eBay items in the provided list using a ThreadPoolExecutor.

    Args:
        search_results (List[eBayProduct]): A list of eBayProduct objects to process images for.
    """
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_ebay_item_image, item) for item in search_results]
        
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as e:
                print(f"An error occurred: {e}")


async def process_item_image(item: eBayProduct) -> None:
    """
    Asynchronously processes an item's image by fetching and encoding it.

    Args:
        item (eBayProduct): The item whose image is to be processed.
    """
    await item.async_process_images()


async def process_ebay_images_with_async(search_results: List[eBayProduct]) -> None:
    """
    Asynchronously processes images for all items in the search results.

    Args:
        search_results (List[eBayProduct]): A list of items to process images for.
    """
    tasks = [process_item_image(item) for item in search_results]
    await asyncio.gather(*tasks)


def download_and_save_images(image_urls: List[str]) -> List[str]:
    """
    Downloads and saves images from a list of URLs.

    Args:
        image_urls (List[str]): A list of URLs of the images to be downloaded.

    Returns:
        List[str]: A list of paths where the images were saved.
    """
    image_paths = []
    for url in image_urls:
        filename = url.split("/")[-1]
        local_file_path = Path(f"{base_abs_path}/data/{filename}")

        local_file_path.parent.mkdir(parents=True, exist_ok=True)
        # Perform the GET request
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            # Write the content of the response to a local file
            with open(local_file_path, "wb") as image_file:
                image_file.write(response.content)
                image_paths.append(str(local_file_path))
            print(
                f"Image file downloaded successfully and saved as '{local_file_path}'."
            )
        else:
            print(f"Failed to download file. Status code: {response.status_code}")

    return image_paths


def plot_images(image_paths: list, output_path: Optional[str] = None) -> str:
    """
    Plots images from given paths and optionally saves them. Returns the plot as a base64 encoded string.

    Args:
        image_paths (list): A list of paths to the images to be plotted.
        output_path (Optional[str]): The path where the output image will be saved as a .jpg file. If None, the image is not saved.

    Returns:
        str: Base64 encoded string of the plot image.
    """
    images_shown = 0
    plt.figure(figsize=(16, 12))
    for img_path in image_paths:
        if os.path.isfile(img_path):
            image = PILImage.open(img_path)

            plt.subplot(5, 5, images_shown + 1)
            plt.imshow(image)
            plt.xticks([])
            plt.yticks([])

            images_shown += 1
            if images_shown >= 25:
                break

    if output_path:
        plt.savefig(output_path, format="jpg")

    # Save plot to a BytesIO stream and then encode it as base64
    buf = BytesIO()
    plt.savefig(buf, format="jpg")
    buf.seek(0)
    base64_img = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()  # Close the figure to free memory
    return base64_img


def create_pdf(image_path: str, text: str, research: str, content: str, base_dir: str = '.') -> str:
    """
    Generate a PDF from HTML content and save it to a named temporary file.

    Args:
        image_path (str): Path to the image file to include in the PDF.
        text (str): Text content to include in the PDF.
        research (str): Research content to include in the PDF.
        content (str): Additional content to include in the PDF.
        base_dir (str): Base directory to save the PDF file. Defaults to current directory.

    Returns:
        str: The full file path to the generated PDF.
    """
    text_html = markdown.markdown(text)
    research_html = markdown.markdown(research)
    content_html = markdown.markdown(content)
    
    image_path_full = f"file:///{image_path}"

    # style_path = os.path.join(base_dir, 'src', 'style.css')
    html_content = f"""
        <html>
        <head>
        </head>
        <body>
            <img src="{image_path_full}" alt="Cover Image" class="cover-image">
            {text_html}
            {research_html}
            {content_html}
        </body>
        </html>
        """

    temp_dir = os.path.join(base_dir, 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=temp_dir, delete=False, suffix=".pdf") as tmpfile:
        pdf_path = tmpfile.name

    # base_url = os.path.join(base_dir, 'temp')
    html = HTML(string=html_content, base_url=base_rel_path)
    css = CSS(filename="src/style.css", base_url=base_rel_path)
    html.write_pdf(pdf_path, stylesheets=[css])

    return pdf_path


def base64_to_image(base64_image: str, base_dir: str = '.') -> str:
    """
    Convert a base64 encoded image to a temporary image file and return its full file path.

    Args:
        base64_image (str): The base64 encoded image string.
        base_dir (str): Base directory to save the temporary image file. Defaults to current directory.

    Returns:
        str: The full file path to the temporary image file.
    """
    image_data = base64.b64decode(base64_image)
    image = PILImage.open(io.BytesIO(image_data))
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    
    temp_dir = os.path.join(base_dir, 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=temp_dir, delete=False, suffix=".jpg") as tmpfile:
        image_path = tmpfile.name
        image.save(image_path)
    
    return image_path



