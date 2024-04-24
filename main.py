from concurrent.futures import ThreadPoolExecutor
import tempfile
from typing import List
from urllib.parse import urlparse
import instructor
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

from src.utils.output import write_md_to_pdf

load_dotenv()
import openai
from metaphor_python import Metaphor

# from PIL import Image
from PIL import Image as PILImage
import markdown
import textwrap
from html2text import html2text
from bs4 import BeautifulSoup

from weasyprint import HTML, CSS
from weasyprint.text.fonts import FontConfiguration


st.set_page_config(
    page_title="PropertyImageInspect", layout="centered", initial_sidebar_state="auto"
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



    def display_search(search_results):
        """
        Display the search results.

        Args:
            search_results (list): List of search results.
        """
        for result in search_results:
            st.write(f"Title: {result.title}")
            st.write(f"URL: {result.url}")
            st.write(f"Published Date: {result.published_date}")
            st.markdown("___")

    def display_content(content_results):
        """
        Display internet content.

        Args:
            search_results (list): List of search results.
        """
        for result in content_results:
            st.write(f"Title: {result.title}")
            st.write(f"URL: {result.url}")
            st.markdown("**Content:**")
            st.markdown(f"{result.extract}", unsafe_allow_html=True)
            st.markdown("___")

    def get_page_contents(search_results):
        contents_response = metaphor.get_contents(search_results)
        return contents_response.contents

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

    def create_web_content_string(search_contents: list, char_limit: int = 9000) -> str:
        """
        Build context for LLM call.

        Args:
            search_contents (list): List of search contents.
            char_limit (int, optional): Total character limit. Defaults to 9000.

        Returns:
            str: Processed internet content.
        """
        total_chars = sum(
            [len(clean_html_content(item.extract)) for item in search_contents]
        )
        internet_content = ""

        for item in search_contents:
            cleaned_content = clean_html_content(item.extract)
            item_chars = len(cleaned_content)
            slice_ratio = item_chars / total_chars
            slice_limit = int(char_limit * slice_ratio)
            sliced_content = cleaned_content[:slice_limit]

            internet_content += f"--START ITEM--\nURL: {item.url}\nTITLE: {item.title}\nCONTENT: {sliced_content}\n--END ITEM--\n"

        return internet_content

    def format_for_markdown(text: str) -> str:
        """
        Formats the given text for markdown.

        Args:
            text (str): The text to be formatted.

        Returns:
            str: The formatted text.
        """
        # Split the text into items
        items = text.split("--END ITEM--")

        # Process each item
        formatted_items = []
        for item in items:
            if item.strip() == "":
                continue

            # Remove START ITEM tag and split into lines
            lines = item.replace("--START ITEM--", "").strip().split(" ")

            # Initialize formatted item
            formatted_item = "\n\n"

            # Add each line with a newline at the end
            for line in lines:
                if "URL:" in line or "TITLE:" in line:
                    formatted_item += "\n" + line
                elif "CONTENT:" in line:
                    formatted_item += "\n" + line + "\n"
                else:
                    formatted_item += " " + line

            formatted_items.append(formatted_item.strip())

        return "\n\n".join(formatted_items)


    def synthesize_report(topic: str, internet_content: str) -> str:
        full_report = ""
        for completion in openai.chat.completions.create(
            model="gpt-4-1106-preview",
            temperature=1,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful internet research assistant specializing in empowering buyers. You help sift through raw search results to find the most relevant and interesting findings for user topic of interest.",
                },
                {
                    "role": "user",
                    "content": "Input Data:\n"
                    + internet_content
                    + f"Write a two paragraph research report about **{topic}** based on the provided search results. One paragraph summarizing the Input Data, and another focusing on the main Research Topic. Include as many sources as possible. ALWAYS cite results using [[number](URL)] notation after the reference. End with a markdown table of all the URLs used. Remember to use markdown links when citing the context, for example [[number](URL)].",
                },
            ],
            stream=True,
        ):
            # Check if there is content to display
            if completion.choices[0].delta.content is not None:
                full_report += completion.choices[0].delta.content
                report_placeholder.markdown(full_report + "▌")
        # Final update to placeholder after the stream ends
        report_placeholder.markdown(full_report)
        return full_report

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
        additional_details = ui.textarea(
            default_value="Add any additional details or context about the image here:",
            # disabled=not show_details,
            # height=125,
        )


    # Add a radio button for the prompt type
    # prompt_type = st.radio(
    #     "Choose the type of item for analysis:",
    #     ("Single Image", "Multi Image"),
    # )

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
                    "Understand that users may inadvertently provide duplicate images. Simply advise them of this and proceed the analysis. "
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
                    
                    # ====== Web search starts here
                    
                    # TODO: Replace this with image embedding version. Need to get top 3-5 for each query, render and save for later
                    
                    base_queries = generate_base_queries(full_response)
                    
                    cols = st.columns(len(base_queries.search_tasks))
                    
                    for i, search in enumerate(base_queries.search_tasks):
                        with cols[i]:
                            search_string = str(search.title())
                            st.markdown(f"**{search_string}**")
                            averagePrice = AverageSalePrice(
                                query=search_string, 
                                country='us', 
                                condition='all',
                            )
                            st.markdown(str(averagePrice))
                            
                            st.markdown("___")
                        
                        with cols[i]:  
                            sold_items = eBayWebSearch(search_string, alreadySold=True)
                            st.markdown(str(sold_items[0]))
                            st.markdown(str(sold_items[1]))
                            st.markdown(str(sold_items[2]))
                    
                    
                    st.markdown("___")

                    # Define the placeholder for the report outside the container
                    report_placeholder = st.empty()

                    st.markdown("___")

 
                    # The final report will be displayed outside the container
                    report_placeholder.markdown()

                    st.session_state["last_processed_message"] = messages
                    st.session_state["last_full_response"] = full_response

                    st.session_state.analyze_button_clicked = False
                    # Create the PDF
                    st.markdown("___")
                    # Provide the download button
                except Exception as e:
                    st.error(f"An error occurred: {e}")

            else:
                pdf_path = create_pdf(
                    # image_path=image_path,
                    text=st.session_state["last_full_response"],
                    research=" ",
                    content=" ",
                )
                # Display the last stored response
                message_placeholder = st.empty()
                message_placeholder.markdown(st.session_state["last_full_response"])
                st.info(
                    "Displaying previous report as the image and details have not changed."
                )



