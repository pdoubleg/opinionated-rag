from concurrent.futures import ThreadPoolExecutor
import tempfile
import streamlit as st
import os
import io
import re
import base64
import time
from dotenv import load_dotenv

from src.utils.output import write_md_to_pdf

load_dotenv()
import openai
from openai import OpenAI
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


# Initialize session state for last API key
if "last_api_key" not in st.session_state:
    st.session_state["last_api_key"] = ""

# Create a sidebar for API key configuration and additional features
st.sidebar.header("Configuration")
api_key_ = st.sidebar.text_input(
    "Enter your OpenAI API key",
    type="password",
    help=f"Get an api key at www.openai.com",
)

meta_key_ = st.sidebar.text_input(
    "Enter your Metaphor API key",
    type="password",
    help=f"Get 1k free searches at www.metaphor.systems",
)

if not api_key_ or not meta_key_:
    st.warning("Please input api keys")
    st.stop()

if api_key_ and api_key_ != st.session_state["last_api_key"]:
    st.toast("**Thanks!** We're all set now :+1:")
    st.session_state["last_api_key"] = api_key_

lock = st.secrets["SECRET_API_KEY"]

if api_key_:
    if api_key_ == lock:
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        metaphor = Metaphor(api_key=st.secrets["METAPHOR_API_KEY"])
    else:
        client = OpenAI(api_key=api_key_)
        metaphor = Metaphor(api_key=meta_key_)

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

    def get_llm_response(
        system="You are a helpful assistant.",
        user="",
        temperature=1,
        model="gpt-3.5-turbo",
    ):
        completion = openai.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return completion.choices[0].message.content

    def create_base_item_generation_prompt(vision_output):
        return f"""The following is a detailed description of a user-supplied image of an item they want to buy. Take the description, and distill it down to a concise, but descriptive internet search query that will help empower them as a buyer:\n\nITEM_ANALYSIS:\n{vision_output}\n\nHELPFUL_ANSWER: """

    def generate_base_query(vision_output):
        user_prompt = create_base_item_generation_prompt(vision_output)
        completion = get_llm_response(
            system="The user will ask you to help generate a search query. Respond with only the query in plain text with no extra formatting.",
            user=user_prompt,
            temperature=0.3,
        )
        return completion

    def create_keyword_query_generation_prompt(topic, n):
        return f"""I'm conducting research on {topic} and need help coming up with Google keyword search queries.
    Google keyword searches should just be a few words long. It should not be a complete sentence.
    Please generate a diverse list of {n} Google keyword search queries that would be useful for researching ${topic}. Do not add any formatting or numbering to the queries."""

    def generate_search_queries(topic, n):
        user_prompt = create_keyword_query_generation_prompt(topic, n)
        completion = get_llm_response(
            system="The user will ask you to help generate some search queries. Respond with only the suggested queries in plain text with no extra formatting, each on it's own line.",
            user=user_prompt,
            temperature=1,
        )
        queries = [s for s in completion.split("\n") if s.strip()][:n]
        return queries

    def get_keyword_search_results(queries, linksPerQuery=1):
        results = []
        for query in queries:
            search_response = metaphor.search(
                query, type="keyword", num_results=linksPerQuery, use_autoprompt=False
            )
            results.extend(search_response.results)
        return results

    def get_neural_search_results(queries, linksPerQuery=1):
        results = []
        for query in queries:
            search_response = metaphor.search(
                query, type="neural", num_results=linksPerQuery, use_autoprompt=True
            )
            results.extend(search_response.results)
        return results

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

    # File uploader allows user to add their own image
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        # Display the uploaded image
        with st.expander("Image", expanded=True):
            st.image(uploaded_file, caption=uploaded_file.name, use_column_width=True)

    # Create two columns for the toggles
    col1, col2, col3 = st.columns(3)

    # Toggle for showing additional details input in the first column
    with col1:
        show_details = st.toggle("Add details about the image", value=False)

        if show_details:
            # Text input for additional details about the image, shown only if toggle is True
            additional_details = st.text_area(
                "Add any additional details or context about the image here:",
                disabled=not show_details,
                height=125,
            )

    # Toggle for including audio in the second column
    with col2:
        include_audio = st.toggle("Include audio report", value=False)

        if include_audio:
            # Radio button for selecting audio type, shown only if toggle is True
            audio_type = st.radio(
                "Choose the type of audio:",
                ("alloy", "echo", "fable", "onyx", "nova", "shimmer"),
                index=4,
            )

    # Radio button for selecting web search type
    with col3:
        web_search_type = st.radio(
            "Choose the type of web search:",
            ("keyword", "neural", "none"),
            index=0,
            help="`keyword` runs a standard search - good for specific topics\n\n`neural` predicts interesting links - good when a topic is popularly discussed online\n\nLearn more at: https://docs.metaphor.systems/reference/prompting-guide",
        )

    # Add a radio button for the prompt type
    prompt_type = st.radio(
        "Choose the type of item for analysis:",
        ("Personal Property", "Structural Items", "Diagram Inspector"),
    )

    # Button to trigger the analysis
    analyze_button = st.button("Inspect the Image", type="secondary")

    if analyze_button:
        st.session_state.analyze_button_clicked = True

    # Check if an image has been uploaded, if the API key is available, and if the button has been pressed
    if uploaded_file is not None and st.session_state.analyze_button_clicked:
        with st.spinner("Inspecting the image ..."):
            # Encode the image
            base64_image = encode_image(uploaded_file)

            # Save the uploaded image to a file
            image_path = base64_to_image(base64_image)
            with open(image_path, "wb") as img_file:
                img_file.write(uploaded_file.getvalue())

            # Set the prompt text based on the selected type
            if prompt_type == "Personal Property":
                prompt_text = (
                    "You are a highly knowledgeable eBay power seller. "
                    "Your task is to examine the following image in detail. "
                    "Begin with a descriptive title caption in bold. "
                    "Provide a comprehensive, factual, and price-focused explanation of what the image depicts. "
                    "Highlight key elements and their significance, and present your analysis in clear, well-structured markdown format. "
                    "If applicable, include any relevant facts to enhance the explanation. "
                    "TITLE: "
                )

            elif prompt_type == "Structural Items":
                prompt_text = (
                    "You are a highly knowledgeable structural damage appraiser. "
                    "Your task is to generate a highly concise description of the image focusing on repairs needed. "
                    "Your task is to examine the following image in detail. "
                    "Begin with a descriptive title caption in bold. "
                    "Provide a comprehensive, scope-of-repair-focused explanation of what the image depicts. "
                    "Itemize key repair scope items, and present your analysis in clear, well-structured markdown format. "
                    "If applicable, include any relevant considerations such as demolition, or mitigation to enhance the scope explanation. "
                    "If possible, include rough approximations of time and material for each scope item. "
                    "TITLE: "
                )

            elif prompt_type == "Diagram Inspector":
                prompt_text = (
                    "You are a highly knowledgeable diagram interpreter specializing in deciphering analytical plots and technical diagrams. "
                    "Your task is to examine the following image in detail. "
                    "Begin with a descriptive title caption in bold. "
                    "Provide a comprehensive and clear explanation of what the image depicts. "
                    "Itemize key observations, and present your analysis in clear, well-structured markdown format. "
                    "If applicable, include any relevant considerations such as the image's context or motivations. "
                    "If possible, include rough approximations of any quantifyable features of the image. "
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
                        {
                            "type": "image_url",
                            "image_url": f"data:image/jpeg;base64,{base64_image}",
                        },
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

                    if include_audio:
                        with st.spinner(text="Please hold for audio..."):
                            response = client.audio.speech.create(
                                model="tts-1",
                                voice=audio_type,
                                input=full_response,
                            )
                            # Adjusted to use with_streaming_response for proper streaming
                            with response.with_streaming_response() as streaming_response:
                                with open("output.mp3", "wb") as audio_file:
                                    for chunk in streaming_response.iter_bytes():
                                        audio_file.write(chunk)
                            audio_file = open("output.mp3", "rb")
                            audio_bytes = audio_file.read()
                            st.success("Audio is Ready")
                        st.audio(audio_bytes)

                    st.markdown("___")

                    if web_search_type !="none":
                        st.markdown("## Internet Research:")

                        item_title = generate_base_query(full_response)

                        search_queries = generate_search_queries(item_title, 3)
                        st.markdown("Running suggested searches:")
                        st.markdown(f"* {search_queries[0]}")
                        st.markdown(f"* {search_queries[1]}")
                        st.markdown(f"* {search_queries[2]}")

                        with st.status(
                            "Links to suggested research material", state="running"
                        ) as status:
                            if web_search_type == "keyword":
                                search_result_links = get_keyword_search_results(
                                    search_queries, 2
                                )
                            else:
                                search_result_links = get_neural_search_results(
                                    search_queries, 2
                                )
                            display_search(search_result_links)
                            status.update()

                        st.markdown("___")

                        # Define the placeholder for the report outside the container
                        report_placeholder = st.empty()

                        st.markdown("___")

                        with st.status(
                            "Synthesizing internet search results...", state="running"
                        ) as status:
                            search_result_contents = get_page_contents(
                                [link.id for link in search_result_links]
                            )
                            internet_content = create_web_content_string(
                                search_result_contents, 30000
                            )
                            full_report = synthesize_report(item_title, internet_content)
                            formatted_content = format_for_markdown(internet_content)
                            html_content = display_content(search_result_contents)
                            st.markdown(f"{html_content}", unsafe_allow_html=True)
                            status.update(label="Source content")

                        # The final report will be displayed outside the container
                        report_placeholder.markdown(full_report)

                        # Strip all URLs from the full_report string
                        full_report_audio = re.sub(r"http\S+", "", full_report)

                        if include_audio:
                            with st.spinner(text="Please hold again..."):
                                response_ = client.audio.speech.create(
                                    model="tts-1",
                                    voice=audio_type,
                                    input=full_report_audio,
                                )
                                # Adjusted to use with_streaming_response for proper streaming
                                with response_.with_streaming_response() as streaming_response_:
                                    with open("report.mp3", "wb") as audio_file_:
                                        for chunk in streaming_response_.iter_bytes():
                                            audio_file_.write(chunk)
                                audio_file_ = open("report.mp3", "rb")
                                audio_bytes_ = audio_file_.read()
                            st.success("Audio is Ready")
                            st.audio(audio_bytes_)

                        st.session_state["last_processed_message"] = messages
                        st.session_state["last_full_response"] = full_response

                        st.session_state.analyze_button_clicked = False
                        # Create the PDF
                    #     pdf_path = create_pdf(
                    #         image_path,
                    #         full_response,
                    #         full_report,
                    #         internet_content,
                    #     )
                    # else:
                    #     pdf_path = create_pdf(
                    #             image_path=image_path,
                    #             text=full_response,
                    #             research=" ",
                    #             content=" ",
                    #         )
                    # with st.expander:  
                    #     show_pdf(pdf_path)
                    st.markdown("___")
                    # Provide the download button
                    # with open(pdf_path, "rb") as pdf_file:
                    #     PDFbyte = pdf_file.read()

                    # st.download_button(label="Export_Report",
                    #                     data=PDFbyte,
                    #                     file_name="report.pdf",
                    #                     mime='application/octet-stream')
                    

                    # with open(st.session_state.pdf_path, "rb") as file:
                    #     btn = st.download_button(
                    #         label="Download Report & Reset App",
                    #         data=file,
                    #         file_name="report.pdf",
                    #         mime="application/pdf",
                    #     )

                    # Display the response in the app
                    # st.write(response.choices[0].message.content)
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



    else:
        # Warnings for user action required
        if not uploaded_file and analyze_button:
            st.warning("Please upload an image.")
