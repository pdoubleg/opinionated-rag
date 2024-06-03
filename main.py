import asyncio
from concurrent.futures import ThreadPoolExecutor
import tempfile
from typing import List, Tuple
from urllib.parse import urlparse
import instructor
import lancedb
import marvin
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
import streamlit as st
import streamlit_shadcn_ui as ui
from streamlit_extras.add_vertical_space import add_vertical_space as avs
import os
import io
import re
import base64
import time
from dotenv import load_dotenv
from tenacity import Retrying, stop_after_attempt, wait_fixed
from src.doc_store.ebay_scraper import AverageSalePrice, eBayWebSearch
from src.doc_store.ebay_utils import deduplicate_products, eBayItems, process_ebay_images_with_async, process_ebay_images_with_threadpool
from src.doc_store.ebay_scraper import eBayProduct

from src.utils.output import format_message_string, write_md_to_pdf

load_dotenv()
import openai
from marvin.beta.assistants import Assistant, Thread
from marvin.beta.assistants.formatting import pprint_messages, pprint_run

# from PIL import Image
from PIL import Image as PILImage
import markdown
import textwrap
from html2text import html2text
from bs4 import BeautifulSoup

from weasyprint import HTML, CSS
from weasyprint.text.fonts import FontConfiguration

from typing import Callable
from langchain_openai import ChatOpenAI

from src.search.threadpool import run_functions_tuples_in_parallel
from src.utils import llm


st.set_page_config(
    page_title="PropertyImageInspect", layout="wide", initial_sidebar_state="auto"
)

# Streamlit page setup
st.title("`πi` | PropertyImageInspector\n\nPowered by: `GPT-4 Turbo with Vision`")


class eBayQueryList(BaseModel):
    """A eBay item that is being searched."""
    
    item_name: str = Field(
        ...,
        description="A concise, but descriptive name for the item of interest."
    )
    search_queries: List[str] = Field(
        default_factory=list,
        description="A diverse list of alternative eBay search queries that compliment the primary item of interest."
    )
    
    @property
    def search_tasks(self):
        alternates = [query for query in self.search_queries]
        search_tasks = [str(self.item_name)]
        search_tasks.extend(alternates)
        return search_tasks


def search_image_vectors(image_path: str, df: pd.DataFrame, n: int = 10) -> List[eBayItems]:
    # unique_table_name = f"ebay_{uuid.uuid4().hex}"
    unique_table_name = "test_table"
    db = lancedb.connect("./.lancedb")
    table = db.create_table(unique_table_name, schema=eBayItems, mode='overwrite')
    table.add(df)
    query_image = PILImage.open(image_path)
    res = table.search(query_image) \
        .limit(n) \
        .to_pydantic(eBayItems)
    # db.drop_table(unique_table_name)
    return res
      
        
def get_similar_listings(search_query: str, image_path: str, n: int = 5, sold: bool=True) -> List[eBayItems]:
    """
    Searches for similar eBay listings based on a query and an image.

    Args:
        search_queries (List[str]): A list of search queries.
        image_path (str): The path to the image for similarity search.
        n (int): Number of image results per search query after sorting by image similarity.

    Returns:
        List[eBayItems]: A list of the best matching listings based on image similarity.
    """
    st.toast(f"Searching: {search_query}")
    search_results = eBayWebSearch(search_query, alreadySold=sold)
    process_ebay_images_with_threadpool(search_results)
    search_results = deduplicate_products(search_results)
    st.markdown(f"Top **{n}** matches based on analysis of **{len(search_results):,}** listings:")
    df = pd.DataFrame([r.to_data_dict for r in search_results])
    best_images = search_image_vectors(image_path, df, n)

    return best_images


USEFUL_PAT = "Yes useful"
NONUSEFUL_PAT = "Not useful"

image_comp_prompt = """
I'm doing some pricing research for the first image and looking for good 'comps' USEFUL as comparison point.  
Please review and asses whether the second image is similar enough to be USEFUL. Respond with EXACTLY AND ONLY: "Yes useful" or "Not useful"
"""

def llm_eval_chunk(target_image: str, comp_image: str) -> bool:  

    def _extract_usefulness(model_output: str) -> bool:
        """Default 'useful' if the LLM doesn't match pattern exactly.
        This is because it's better to trust the (re)ranking if LLM fails"""
        if model_output.strip().strip('"').lower() == NONUSEFUL_PAT.lower():
            return False
        return True

    model_output = llm.prompt_multi_image_input(
        prompt=image_comp_prompt,
        image_paths=[
            target_image,
            comp_image,
        ]
    )

    return _extract_usefulness(model_output)


def llm_batch_eval_chunks(
    target_image: str, comp_images: list[str], use_threads: bool = True
) -> list[bool]:
    if use_threads:
        functions_with_args: list[tuple[Callable, tuple]] = [
            (llm_eval_chunk, (target_image, comp_image)) for comp_image in comp_images
        ]

        print(
            "Running LLM usefulness eval in parallel (following logging may be out of order)"
        )
        parallel_results = run_functions_tuples_in_parallel(
            functions_with_args, allow_failures=True
        )

        # In case of failure/timeout, don't throw out the chunk
        return [True if item is None else item for item in parallel_results]

    else:
        return [
            llm_eval_chunk(target_image, comp_image) for comp_image in comp_images
        ]


def filter_chunks(
    target_image: str,
    chunks_to_filter: list[eBayItems | eBayProduct],
    max_llm_filter_chunks: int = 20,
) -> list[eBayItems | eBayProduct]:
    """Filters chunks based on whether the LLM thought they were relevant to the query.

    """
    if isinstance(chunks_to_filter[0], eBayItems):
        chunks_to_filter = chunks_to_filter[: max_llm_filter_chunks]
        llm_chunk_selection = llm_batch_eval_chunks(
            target_image=target_image,
            comp_images=[chunk.image_url for chunk in chunks_to_filter],
        )
        return [
            chunk
            for ind, chunk in enumerate(chunks_to_filter)
            if llm_chunk_selection[ind]
        ]
    else:
        chunks_to_filter = chunks_to_filter[: max_llm_filter_chunks]
        llm_chunk_selection = llm_batch_eval_chunks(
            target_image=target_image,
            comp_images=[str(chunk.item.image_url) for chunk in chunks_to_filter],
        )
        return [
            chunk
            for ind, chunk in enumerate(chunks_to_filter)
            if llm_chunk_selection[ind]
        ]
        

def thumbnail(image, scale=3):
    return image.resize(np.array(image.size)//scale)


def upscale_image(image: PILImage.Image, scale: int = 3) -> PILImage.Image:
    """
    Increase the size of an image while preserving its quality.

    Args:
        image (Image.Image): The input image to be upscaled.
        scale (int): The factor by which to scale the image. Default is 3.

    Returns:
        Image.Image: The upscaled image.
    """
    new_size = np.array(image.size) * scale
    return image.resize(new_size, PILImage.LANCZOS)



def render_search_results(
    all_results: List[Tuple[List, List]], 
    search_query_list: List[str], 
    query_number: int, 
    search_type_code: int):
    search_type_dict = {
        '0': "eBay's search engine",
        '1': 'Image Analysis',
    }
    search_type = search_type_dict.get(str(search_type_code), None)
    query = search_query_list[query_number]
    base_results = all_results[query_number][0]
    results = all_results[query_number][search_type_code]
    st.markdown((f"Search for `{query}` returned {len(base_results)} results"))
    st.markdown((f"Top results based on **{search_type}**:\n"))
    for result in results:
        st.image(thumbnail(result.image))
        st.markdown((str(result)))


# Initialize session state for last API key
if "last_api_key" not in st.session_state:
    st.session_state["last_api_key"] = ""

# Create a sidebar for API key configuration and additional features
st.sidebar.header("Configuration")
api_key_ = st.sidebar.text_input(
    "Enter your email address",
)

ADMIN_USERS = {
    'pdoubleg@gmail.com',
    'test@example.com',
    'person3@email.com'
}

user_dict = st.experimental_user.to_dict()

if user_dict['email'] or api_key_ in ADMIN_USERS:

    # Function to encode the image to base64
    def encode_image(image_file):
        return base64.b64encode(image_file.getvalue()).decode("utf-8")

    with st.expander(":camera:**Introduction**"):
        st.markdown(
            """
            ## Welcome to Property Image Inspector!
            
            This app uses GPT-4 Turbo with Vision to inspect images, and Metaphor API to research the internet.\n\n 
            Here's how it works:
            
            * Upload an image
            * Tell the app if your image contains "Personal Property", "Structural Items", or a diagram/plot
            * Optionally:
              - Add instructions, or more info about the image
              - Get an audio version of the report
              - Specify your web search preference
            * Click the 'Inspect the Image' button to get a
              - Detailed image analysis
              - Curated internet content summarized as a research report with citations
              - Optional PDF report
            * That's it :sunglasses:
        """
        )

    # Initialize session state for app re-run
    if "analyze_button_clicked" not in st.session_state:
        st.session_state["analyze_button_clicked"] = False

    st.session_state["analyze_button_clicked"] = False

    # Initialize session state for last processed message
    if "last_processed_message" not in st.session_state:
        st.session_state["last_processed_message"] = None

    if "last_full_response" not in st.session_state:
        st.session_state["last_full_response"] = ""

    if "pdf_path" not in st.session_state:
        st.session_state["pdf_path"] = ""

    def generate_base_queries(
        item_description: str, 
    ) -> eBayQueryList:
        client = instructor.from_openai(openai.OpenAI())
        return client.chat.completions.create(
            model="gpt-4-turbo",
            temperature=0.3,
            response_model=eBayQueryList,
            max_retries=Retrying(
                stop=stop_after_attempt(5),
                wait=wait_fixed(1),
            ),
            messages=[
                {
                    "role": "system",
                    "content": "You are a world class eBay assistant.",
                },
                {
                    "role": "user",
                    "content": f"The following is a detailed description of an item the user wants to buy on eBay. Take the description, and distill it down to concise, but descriptive name along with **2** diverse search queries for the item::\n\nITEM DESCRIPTION: {item_description}",
                },
            ],
            stream=False,
        )



    def clean_html_content(content: str) -> str:
        """
        Clean the HTML content using BeautifulSoup.

        Args:
            content (str): HTML content.

        Returns:
            str: Cleaned text content.
        """
        soup = BeautifulSoup(content, "html.parser")

        # Extract header and paragraph tags
        header_tags = soup.find_all(re.compile(r"^h\d$"))
        paragraph_tags = soup.find_all("p")

        # Strip HTML tags and collect text content
        stripped_content = ""
        for tag in header_tags + paragraph_tags:
            stripped_content += " " + tag.get_text().strip() + " "

        return " ".join(stripped_content.split())

    # Generate PDF report with the uploaded image and model output
    def markdown_to_text(markdown_string):
        # Convert markdown to html
        html = markdown.markdown(markdown_string)
        # Convert html to plain text while preserving line breaks
        soup = BeautifulSoup(html, features="html.parser")
        text = ""
        for string in soup.stripped_strings:
            # Wrap the text after 80 characters
            wrapped_string = textwrap.fill(string, width=90)
            text += wrapped_string + "\n"
        return text

    def normalize_image_size(image_path, max_width, max_height):
        with PILImage.open(image_path) as img:
            width, height = img.size
            aspect_ratio = width / height

            if width > max_width:
                width = max_width
                height = width / aspect_ratio

            if height > max_height:
                height = max_height
                width = height * aspect_ratio

            return int(width), int(height)

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
                <link rel="stylesheet" type="text/css" href="src/style.css">
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
        html = HTML(string=html_content)
        css = CSS(filename="src/style.css")
        html.write_pdf(pdf_path, stylesheets=[css])

        return pdf_path

    # Convert base64 to file format
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
    
    def show_pdf(file_path):
        with open(file_path,"rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="800" height="800" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)


    def is_url(image_path: str) -> bool:
        """
        Check if the given string is a valid URL.

        Args:
            image_path (str): The string to check.

        Returns:
            bool: True if the string is a valid URL, False otherwise.
        """
        try:
            result = urlparse(image_path)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False
    
    def encode_image_to_base64(image_path: str) -> str:
        """
        Encode a local image file to a base64 string.

        Args:
            image_path (str): The path to the image file.

        Returns:
            str: The base64 encoded string of the image.
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    
    # File uploader allows user to add their own images
    uploaded_files = st.file_uploader("Upload one or more images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

    if uploaded_files:
        # Display the uploaded images
        with st.expander("Images", expanded=True):
            for uploaded_file in uploaded_files:
                st.image(uploaded_file, caption=uploaded_file.name, use_column_width=True)

    show_details = st.toggle("Add details about the image", value=False)

    if show_details:
        # Text input for additional details about the image, shown only if toggle is True
        additional_details = st.text_area(
            label="Additional details",
            disabled=not show_details,
            height=125,
        )


    # Button to trigger the analysis
    analyze_button = ui.button('Launch Inspection', key="clk_btn", className="bg-green-950 text-white")
    # analyze_button = st.button("Inspect the Image", type="secondary")

    if analyze_button:
        st.session_state.analyze_button_clicked = True

    # Check if an image has been uploaded, if the API key is available, and if the button has been pressed
    if st.session_state.analyze_button_clicked:
        with st.spinner("Inspecting the image ..."):
            # Encode the image
            base64_images = [encode_image(file) for file in uploaded_files]

            # Save the uploaded images to files
            image_paths = []
            for i, base64_img in enumerate(base64_images):
                image_path = base64_to_image(base64_img)
                image_paths.append(image_path)
                
                with open(image_path, "wb") as img_file:
                    img_file.write(uploaded_files[i].getvalue())
                    
            target_image_path = image_paths[0]

            # Set the prompt text based on number of photos
            
            if len(image_paths) == 1:
                prompt_type = "Single Image"
            else:
                prompt_type = "Multi Image"
            
            if prompt_type == "Single Image":
                prompt_text = (
                    "You are a highly knowledgeable eBay power seller. "
                    "Your task is to examine the following image in detail. "
                    "Begin with a descriptive title caption using level one markdown heading. "
                    "Provide a comprehensive, factual, and price-focused explanation of what the image depicts. "
                    "Highlight key elements and their significance, and present your analysis in clear, well-structured markdown format. "
                    "If applicable, include any relevant facts to enhance the explanation. "
                    "TITLE: "
                )

            elif prompt_type == "Multi Image":
                prompt_text = (
                    "You are a highly knowledgeable eBay power seller. "
                    "Your task is to examine the following set of images in detail, which are all of the same item. "
                    "Begin with a descriptive title caption using level one markdown heading. "
                    "Briefly summarize each of the views and how they contribute to the overall understanding of the item. "
                    "Provide a concise, factual, and price-focused explanation of what the images depict. "
                    "Highlight key elements and their significance, and present your analysis in clear, well-structured markdown format. "
                    "TITLE: "
                )

            if show_details and additional_details:
                prompt_text += f"\n\nAdditional Context or Instructions Provided by the User:\n{additional_details}"

            # Create the payload for the completion request
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                    ] + [
                        {"type": "image_url", "image_url": {"url": path} if is_url(path) else f"data:image/jpeg;base64,{encode_image_to_base64(path)}"}
                        for path in image_paths
                    ],
                }
            ]

            # Check if the current message is different from the last processed message
            if messages != st.session_state["last_processed_message"]:
                # Make the request to the OpenAI API
                try:
                    # Without Stream

                    # response = client.chat.completions.create(
                    #     model="gpt-4-vision-preview", messages=messages, max_tokens=500, stream=False
                    # )
                    
                    client = openai.OpenAI()
                    # Stream the response
                    full_response = ""
                    message_placeholder = st.empty()
                    for completion in client.chat.completions.create(
                        model="gpt-4-vision-preview",
                        messages=messages,
                        max_tokens=1200,
                        stream=True,
                    ):
                        # Check if there is content to display
                        if completion.choices[0].delta.content is not None:
                            full_response += completion.choices[0].delta.content
                            message_placeholder.markdown(full_response + "▌")
                    # Final update to placeholder after the stream ends
                    message_placeholder.markdown(full_response)

                    st.markdown("___")
                                                                              
                    base_queries = generate_base_queries(full_response)
                    st.markdown(f"## Summary for {len(base_queries.search_tasks)} Search Variations")
                    avs(1)
                    
                    status_placeholder = st.progress(0.0, text=" ")
                    
                    cols = st.columns(len(base_queries.search_tasks))
                    
                    for i, search in enumerate(base_queries.search_tasks):
                        status_placeholder.progress(value=i/len(base_queries.search_tasks), text=f"Searching: {search}")
                        with cols[i]:
                            search_string = str(search.title())
                            st.markdown(f"**{search_string}**")
                            averagePrice = AverageSalePrice(
                                query=search_string, 
                                country='us', 
                                condition='all',
                            )
                            price_string = str(averagePrice).replace("\n\n", "\n")
                            price_string = price_string.replace("$", "＄")
                            st.write(price_string)
                            
                            st.markdown("___")
                    
                    status_placeholder.empty()
                    
                    status_placeholder2 = st.progress(0.0, text=" ")
                    
                    research_tabs = st.tabs(base_queries.search_tasks)
                    
                    all_results = []
                    
                    for i, search in enumerate(base_queries.search_tasks):
                        status_placeholder2.progress(value=i/len(base_queries.search_tasks), text=f"Analyzing listings for: {search}")
                        search = base_queries.search_tasks[i]
                        with research_tabs[i]:
                            with st.status(f"Searching for '{search}'", expanded=False) as status:
                                res = get_similar_listings(
                                    search_query=search,
                                    image_path=target_image_path,
                                    n=5,
                                )
                                all_results.extend(res)
                                for i in range(len(res)):
                                        text_col, image_col = st.columns([0.6, 0.4], gap="medium")
                                        with text_col:
                                            st.markdown(str(res[i]))
                                        with image_col:
                                            sized_image = upscale_image(res[i].image, scale=3)
                                            st.image(sized_image)
                                status.update(label=f"Completed research for: {search}", state="complete", expanded=True)
                                                                      
                    st.markdown("___")
                    status_placeholder2.empty()
                    
                    combined_search_results = ""
                    
                    with st.status(f"Getting best matches ...", expanded=False) as status:
                        filtered_chunks = filter_chunks(target_image_path, all_results)
                        st.markdown(f"\nReturned {len(filtered_chunks)} best matching images from {len(all_results)} candidates\n\n")
                        for obj in filtered_chunks:
                                text_col2, image_col2 = st.columns([0.6, 0.4], gap="medium")
                                with text_col2:
                                    st.markdown(str(obj))
                                    combined_search_results += str(obj)
                                    combined_search_results += "\n\n"
                                with image_col2:
                                    sized_image = upscale_image(obj.image, scale=3)
                                    st.image(sized_image)
                        status.update(label=f"{len(filtered_chunks)} Items Selected", state="complete", expanded=True)  
                        st.markdown("_")
                    
                    ai = Assistant(
                        instructions="You are a helpful research assistant, skilled at drafting engaging reports that are well structured and nicely formatted.",
                    )
                    thread = Thread(
                    )
                    thread.add(
                        f"Please analyze the ITEM DESCRIPTION and SEARCH RESULTS and use them to write a market research-style report. \
                        Include in your report details on the top 5 SEARCH RESULTS that most closely resemble the DESCRIPTION. \
                        Note that the SEARCH RESULTS contain a mix of text and image based searches. You can select from ANY of them to find the best matches.\n\nITEM DESCRIPTION: {full_response}\n\nSEARCH RESULTS: {combined_search_results}",
                    )
                    with st.spinner(text=f"Generating research summary ..."):
                        run = thread.run(ai)
                        run_messages = thread.get_messages()
                        query_eval_message = format_message_string(run_messages[-1])
                        eval_string = query_eval_message.replace("$", "＄")
                        st.markdown(eval_string)
                                  
                    st.session_state["last_processed_message"] = messages
                    st.session_state["last_full_response"] = eval_string

                    st.session_state.analyze_button_clicked = False
                    # Create the PDF
                    st.markdown("___")
                    # Provide the download button
                    
                
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                class eBayListingItem(BaseModel):
                    """An item for sale on eBay"""
                    title: str = Field(description="A concise tag-line style title for the item.")
                    item_specifications: str = Field(description="A detailed inspection of the item in the style of product details or technical specs.")
                    seo_style_ebay_listing: str = Field(description="A SEO-focused compelling description of the item. It should be engaging to read, helping users imagine how the item could positively impact their life.")
                    
                    def __str__(self):
                        wrapped_description = textwrap.fill(self.seo_style_ebay_listing, width=100)
                        return f"## {self.title}\n\n**Item Description:**\n\n{wrapped_description}"
                        

                img = marvin.Image.from_path(
                    target_image_path
                )
                result = marvin.cast(
                    data=img, 
                    target=eBayListingItem,
                    instructions="You are a wold class eBay seller, an expert at vividly describing items and crafting irresistible listing descriptions.",
                )
                
                image_test = llm.prompt_image_gen(
                    prompt=str(result.seo_style_ebay_listing),
                    model="dall-e-3",
                    size_category="square",
                    style="vivid",
                )
                
                text_col3, image_col3 = st.columns([0.5, 0.5], gap="medium")
                with text_col3:
                    st.markdown(str(result))
                    
                with image_col3:
                    image_path = image_test.get('file_path')
                    img = PILImage.open(image_path)
                    sized_image = upscale_image(img, scale=2)
                    st.image(sized_image)
                
                with st.expander("PDF Report"):
                    pdf_path = create_pdf(
                        image_path=image_path,
                        text=eval_string,
                        research=combined_search_results,
                        content=" ",
                    )
                    show_pdf(pdf_path)
                




