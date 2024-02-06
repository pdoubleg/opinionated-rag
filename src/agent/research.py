from llm.gpt import GPT_calls
from termcolor import colored
import os
import threading
import queue
import json

# I set the conversation history to 2000 words but you can set it to whatever you want
gpt = GPT_calls(max_words_per_message=100, max_history_words=2000)
gpt_switchboard = GPT_calls(
    json_mode=True, max_history_words=500, stream=False, name="switchboard"
)
gpt_entity_extractor = GPT_calls(json_mode=True, stream=False, name="entity_extractor")


# create a research directory if it doesn't exist
if not os.path.exists("research"):
    os.makedirs("research")


def save_search_results(search_results, search_term, search_type):
    # create an alphanumeric file name from the search term, with underscores between words
    base_file_name = "_".join(
        "".join(e for e in word if e.isalnum()) for word in search_term.split()
    )
    # Initialize the counter for file enumeration
    counter = 1
    # Find the highest existing file number for .txt files
    existing_files = [
        f
        for f in os.listdir("research")
        if os.path.isfile(os.path.join("research", f)) and f.endswith(".txt")
    ]
    if existing_files:
        # Extract the numbers from the existing .txt files and find the maximum
        existing_numbers = [
            int(f.split("_")[0]) for f in existing_files if f.split("_")[0].isdigit()
        ]
        if existing_numbers:  # Ensure there is at least one number to compare
            counter = max(existing_numbers) + 1
    # Now counter has the next available enumeration
    file_name = f"{counter}_{base_file_name}_{search_type}.txt"
    with open(f"research/{file_name}", "w", encoding="utf-8") as file:
        file.write(search_results)


gpt_switchboard.add_message(
    "system",
    f"""    
    Determine if the user is asking you to perform a web search. Otherwise, chat with the user as normal. Be very attentive to when the user is asking for a search or for you to answer.
    if you determine that the user is asking you to perform a web search or you need real time information to answer user's question, then return 'perpexity'
    Only use exa if the user specifically asks you to use exa.
    always respond with a JSON object as follows:
    {{
        perplexity: <boolean>,
        exa: <boolean>,
    }}
    """,
)

gpt_entity_extractor.add_message(
    "system",
    f"""
    Extract the essential entities from the given text. Only extract what is relevant to the user's question.

    Always respond with a JSON object as follows:
    {{
        entity_1: explanation # string
        entity_2: explanation # string
        entity_3: explanation # string
        .... extract as many entities as you can

    }}
    """,
)


def extract_entities():
    while True:
        if not entity_queue.empty():
            search_term, search_results = entity_queue.get()
            entities = gpt_entity_extractor.chat(
                f"can you please extract the essential entities based on this user question: {search_term}  from the text '{search_results}'"
            )
            # Generate a base file name from the search term, with underscores between words
            base_file_name = "_".join(
                "".join(e for e in word if e.isalnum()) for word in search_term.split()
            )
            # Initialize the counter for file enumeration
            counter = 1
            # Find the highest existing file number
            existing_files = [
                f
                for f in os.listdir("research")
                if os.path.isfile(os.path.join("research", f)) and f.endswith(".json")
            ]
            if existing_files:
                # Extract the numbers from the existing files and find the maximum
                existing_numbers = [int(f.split("_")[0]) for f in existing_files]
                counter = max(existing_numbers) + 1
            # Now counter has the next available enumeration
            file_name = f"{counter}_{base_file_name}.json"

            with open(f"research/{file_name}", "w") as file:
                file.write(json.dumps(entities, indent=4))


entity_queue = queue.Queue()
# start the entity extraction thread
threading.Thread(target=extract_entities, daemon=True).start()

save_search = True if input("Save search results? (y/n): ") == "y" else False
print(f"Saving search results: {save_search}")
while True:
    user_input = input("You: 'save' to toggle save, 'combine' to combine entities: ")
    if user_input == "save":
        save_search = not save_search
        print(f"Saving search results: {save_search}")
        continue
    elif user_input == "combine":
        # find all json files in the research directory and combine them into a single json file
        combined_entities = {}
        for file in os.listdir("research"):
            if file.endswith(".json"):
                with open(f"research/{file}", "r") as f:
                    entities = json.load(f)
                    combined_entities.update(entities)
        with open("research/combined_entities.json", "w") as f:
            f.write(json.dumps(combined_entities, indent=4))

        use_combiner_gpt = input("Use GPT to combine entities? (y/n): ")
        if use_combiner_gpt == "y":
            combiner_gpt = GPT_calls(json_mode=True, stream=True, name="combiner GPT")
            combiner_gpt.add_message(
                "system",
                "Combine the entities from the given JSON objects keep all essential entities and their explanations except for duplicates. Always respond with a JSON object",
            )
            combined_entities = combiner_gpt.chat(json.dumps(combined_entities))
            with open("research/GPT_combined_entities.json", "w") as f:
                f.write(json.dumps(combined_entities, indent=4))

        continue

    gpt.add_message("user", user_input)
    # print("\n")
    search_needed = gpt_switchboard.chat(user_input)

    if search_needed["perplexity"]:
        print(colored("using perplexity search", "cyan"))
        # print("\n")
        search_term = gpt.chat(
            "can you please only return a proper web search term for what I am asking for. Search term should accurately reflect what the I am asking for only return the search term and nothing else. If I am asking you to modify the search term or search in a different direction, be adaptive and creative for creating different search terms when asked. exa and perplexity are search engines, dont inlcude them in your search queries",
            should_print=False,
        )
        print(colored(f"Using search term: {search_term}", "blue"))
        perplexity_result = gpt.perplexity_search(search_term)
        gpt.add_message("assistant", perplexity_result)

        if save_search:
            entity_queue.put((search_term, perplexity_result))
            save_search_results(perplexity_result, search_term, "perplexity")

    elif search_needed["exa"]:
        print(colored("using exa search", "cyan"))
        search_term = gpt.chat(
            "can you please only return a proper web search term for what I am asking for. Search term should accurately reflect what the I am asking for only return the search term and nothing else. If I am asking you to modify the search term or search in a different direction, be adaptive and creative for creating different search terms when asked. exa and perplexity are search engines, dont inlcude them in your search queries",
            should_print=False,
        )
        print(colored(f"Using search term: {search_term}", "blue"))
        search_results = gpt.exa_search(search_term, num_results=3)
        # EXA RESULTS CAN BE VERY LONG EVEN WITH ONLY 3 RESULTS, KEEP THAT IN MIND
        joined_results = "\n##############\n".join(
            [result.url + result.text for result in search_results]
        )

        urls = "\n".join(result.url for result in search_results)
        print(f"using these urls:\n {urls}")

        if save_search:
            entity_queue.put((search_term, joined_results))
            save_search_results(joined_results, search_term, "exa")

        gpt.chat(
            f"Please answer the user question: \n\n {user_input} using these search results: \n\n {joined_results} \n\n explain what you found and what wasnt available in the search results"
        )
        gpt.history = gpt.history[:-2] + gpt.history[-1:]

    else:
        gpt.chat(user_input)
