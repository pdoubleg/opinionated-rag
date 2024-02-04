import streamlit as st
from dotenv import load_dotenv
import os
from opinion_retrieval import download_legal_opinions
from opinion_processing import process_opinions
from augmentation import augment_opinion
from database import init_db, save_opinion

# Load environment variables
load_dotenv()

# Initialize database
init_db()

# Streamlit UI
st.title("Legal Opinion Retrieval and Augmented Generation App")

# Form for user input
with st.form("opinion_retrieval_form"):
    query = st.text_input("Enter your query for legal opinions:")
    submit_button = st.form_submit_button("Retrieve Opinions")

if submit_button:
    if query:
        # Retrieve legal opinions based on the query
        opinions = download_legal_opinions(query)
        if opinions:
            processed_opinions = process_opinions(opinions)
            for opinion in processed_opinions:
                # Augment the retrieved opinions
                augmented_opinion = augment_opinion(opinion)
                # Save the augmented opinions to the database
                save_opinion(augmented_opinion)
                st.write(augmented_opinion)
        else:
            st.write("No opinions found for the given query.")
    else:
        st.write("Please enter a query to retrieve legal opinions.")

