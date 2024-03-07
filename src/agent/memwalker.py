import openai
import json
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

from tenacity import retry, stop_after_attempt, wait_exponential


# Defining the Node class
class Node:
    """
    This class represents a node in the memory tree.
    Each node has a key, a summary, and a list of children nodes.
    """

    def __init__(self, key, summary, children=None):
        """
        Initializes a Node with a key, a summary, and optionally a list of children nodes.
        If no list of children is provided, an empty list is used.
        """
        self.key = key  # The key of the node
        self.summary = summary  # The summary of the node
        self.children = (
            children if children is not None else []
        )  # The children of the node

    def to_dict(self):
        """
        Converts the Node and its children to a dictionary.
        If the children are Nodes, it recursively converts them to dictionaries as well.
        """
        if self.children and isinstance(
            self.children[0], Node
        ):  # If the children are Nodes
            children = [
                child.to_dict() for child in self.children
            ]  # Recursively convert each child to a dictionary
        else:  # If the children are not Nodes
            children = self.children  # Use the children as they are
        return {  # Return a dictionary representation of the Node
            "key": self.key,  # The key of the Node
            "summary": self.summary,  # The summary of the Node
            "children": children,  # The children of the Node
        }


# Defining the MemoryTree class
class MemoryTree:
    """
    This class represents a memory tree.
    The memory tree is constructed from a text, using a specified model, character limit, and number of nodes to combine.
    """

    def __init__(
        self, text, model="gpt-3.5-turbo", char_limit=1000, nodes_to_combine=3
    ):
        """
        Initializes a MemoryTree with a text, a model, a character limit, and a number of nodes to combine.
        It then constructs the tree, saves it to a file, and prints status messages.
        """
        self.text = text  # The text to construct the tree from
        self.model = model  # The model to use for summarizing the text
        self.char_limit = char_limit  # The character limit for each segment of the text
        self.nodes_to_combine = (
            nodes_to_combine  # The number of nodes to combine at each level of the tree
        )
        print("Initializing MemoryTree...")  # Print a status message
        self.tree = self.construct_tree()  # Construct the tree
        print("MemoryTree constructed.")  # Print a status message
        self.save_tree_to_file()  # Save the tree to a file
        print("MemoryTree saved to file.")  # Print a status message

    def construct_tree(self):
        """
        Constructs the memory tree.
        It segments the text, summarizes each segment, and builds the tree from the summaries and segments.
        """
        print("Constructing tree...")  # Print a status message
        segments = [
            self.text[i : i + self.char_limit]
            for i in range(0, len(self.text), self.char_limit)
        ]  # Segment the text
        summaries = [
            self.summarize(segment) for segment in segments
        ]  # Summarize each segment
        return self.build_tree(
            summaries, segments
        )  # Build the tree from the summaries and segments

    def build_tree(self, summaries, segments):
        """
        Builds the memory tree from the summaries and segments.
        It creates a node for each summary and segment, and combines nodes until only one remains.
        """
        print("Building tree...")  # Print a status message
        nodes = [
            Node(f"summary_level_1_{i}", summary, [segments[i]])
            for i, summary in enumerate(summaries)
        ]  # Create a node for each summary and segment
        level = 2  # The current level of the tree
        while len(nodes) > 1:  # While there is more than one node
            print(
                f"Grouping nodes, {len(nodes)} nodes remaining..."
            )  # Print a status message
            grouped_nodes = [
                nodes[i : i + self.nodes_to_combine]
                for i in range(0, len(nodes), self.nodes_to_combine)
            ]  # Group the nodes
            nodes = [
                Node(
                    f"summary_level_{level}_{i}",
                    self.summarize(
                        " ".join([f"{node.key}: {node.summary}" for node in group])
                    ),
                    group,
                )
                for i, group in enumerate(grouped_nodes)
            ]  # Create a new node for each group
            level += 1  # Increment the level
        return nodes[0]  # Return the remaining node

    # Implementing exponential backoff with a minimum wait of 2 seconds, a maximum wait of 60 seconds, and a maximum of 10 attempts
    @retry(
        stop=stop_after_attempt(10), wait=wait_exponential(multiplier=2, min=2, max=60)
    )
    def summarize(self, segment):
        """
        Summarizes a segment of text using the model.
        It sends a chat completion request to the model and concatenates the responses.
        """
        response = (
            openai.ChatCompletion.create(  # Send a chat completion request to the model
                model=self.model,  # The model to use
                messages=[  # The messages to send
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": f"Please summarize the following text: {segment}",
                    },
                ],
                temperature=0,  # The randomness of the model's responses
                stream=True,  # Whether to stream the responses
            )
        )
        responses = ""  # The concatenated responses
        for chunk in response:  # For each chunk of the response
            response_content = (
                chunk.get("choices", [{}])[0].get("delta", {}).get("content")
            )  # Get the content of the chunk
            if response_content:  # If there is content
                responses += response_content  # Add the content to the responses
                print(response_content, end="", flush=True)  # Print the content

        print("\n")  # Print a newline
        return (
            responses.strip()
        )  # Return the responses, stripped of leading and trailing whitespace

    def save_tree_to_file(self):
        """
        Saves the memory tree to a file in JSON format.
        """
        print("Saving tree to file...")  # Print a status message
        with open("memory_tree.json", "w") as f:  # Open the file
            json.dump(self.tree.to_dict(), f, indent=4)  # Dump the tree to the file
        print("Tree saved to file.")  # Print a status message


# Defining the Navigation class
class Navigation:
    """
    This class represents a navigation through a memory tree.
    The navigation is initialized with a tree and a model, and can navigate the tree in response to a query.
    """

    def __init__(self, tree, model="gpt-4"):
        """
        Initializes a Navigation with a tree and a model.
        If the tree is a dictionary, it is converted to a Node.
        """
        print("Initializing Navigation...")  # Print a status message
        self.tree = (
            Node(**tree) if isinstance(tree, dict) else tree
        )  # The tree to navigate
        self.model = model  # The model to use for navigating the tree
        self.memory = []  # The memory of visited nodes
        print("Navigation initialized.")  # Print a status message

    def navigate(self, query):
        """
        Navigates the memory tree in response to a query.
        It generates a prompt, asks the model, parses the response, and either goes to a child node, reverts to the parent node, or returns an answer.
        """
        print("Navigating...")  # Print a status message
        node = self.tree  # The current node
        while node.children:  # While the node has children
            print("Generating prompt...")  # Print a status message
            prompt = self.generate_prompt(node, query)  # Generate a prompt
            print("Asking model...")  # Print a status message
            response = self.ask_model(prompt)  # Ask the model
            print("Parsing response...")  # Print a status message
            result_type, result_value = self.parse_response(
                response
            )  # Parse the response
            if result_type == "action":  # If the result is an action
                if result_value == -1:  # If the action is to revert to the parent node
                    print("Reverting to parent node...")  # Print a status message
                    if self.memory:  # If there is a parent node
                        node = self.memory.pop()  # Revert to the parent node
                else:  # If the action is to go to a child node
                    print(
                        f"Going to child node {result_value}..."
                    )  # Print a status message
                    self.memory.append(node)  # Add the current node to the memory
                    node = Node(**node.children[result_value])  # Go to the child node
            else:  # If the result is an answer
                print("Navigation completed.")  # Print a status message
                return result_value  # Return the answer
        print("Navigation completed.")  # Print a status message
        return node.summary  # Return the summary of the node

    def generate_prompt(self, node, query):
        """
        Generates a prompt for the model based on the current node, the query, and the memory of visited nodes.
        If the node is a leaf node, the prompt asks the model to reason about whether it can answer the query.
        If the node is not a leaf node, the prompt asks the model to reason about which child node to go to next or whether to revert to the parent node.
        """
        visited_keys = " ".join(
            [f"{visited_node.key}" for visited_node in self.memory]
        )  # The keys of the visited nodes
        if node.children and isinstance(
            node.children[0], str
        ):  # If the node is a leaf node
            return f"You are a helpful assistant. Given the query '{query}', the following text: {node.children[0]}, and the keys of the visited nodes: {visited_keys}, please reason about whether you can answer the query. After reasoning, please state 'Reason:' followed by your reasoning, and if you can answer the query, please state 'Answer:' followed by your answer. even when you can't answer the questions return it after 'Answer:'. Always start your answer with 'Answer:'."
        else:  # If the node is not a leaf node
            children_summaries = " ".join(
                [
                    f"{child.key}: {child.summary}"
                    for child in (Node(**child_dict) for child_dict in node.children)
                ]
            )  # The summaries of the child nodes
            return f"You are a helpful assistant. Given the query '{query}', the following summaries of the child nodes: {children_summaries}, and the keys of the visited nodes: {visited_keys}, please reason about which child node to go to next or whether to revert to the parent node. After reasoning, please state 'Reason:' followed by your reasoning, and your action in the format 'Action: summary_level_1_5' or 'Action: revert to parent node'. Always start your action with 'Action:'."

    def ask_model(self, prompt):
        """
        Asks the model a question based on the prompt.
        It sends a chat completion request to the model and concatenates the responses.
        """
        response = (
            openai.ChatCompletion.create(  # Send a chat completion request to the model
                model=self.model,  # The model to use
                messages=[  # The messages to send
                    {"role": "system", "content": prompt},
                ],
                temperature=0,  # The randomness of the model's responses
                stream=True,  # Whether to stream the responses
            )
        )
        responses = ""  # The concatenated responses
        for chunk in response:  # For each chunk of the response
            response_content = (
                chunk.get("choices", [{}])[0].get("delta", {}).get("content")
            )  # Get the content of the chunk
            if response_content:  # If there is content
                responses += response_content  # Add the content to the responses
                print(response_content, end="", flush=True)  # Print the content
        print("\n")  # Print a newline
        return (
            responses.strip()
        )  # Return the responses, stripped of leading and trailing whitespace

    def parse_response(self, response):
        """
        Parses a response from the model.
        If the response contains an answer, it extracts the answer.
        If the response contains an action, it parses the action and extracts the index or the command to revert to the parent node.
        """
        # Check if the response contains an answer
        if "Answer: " in response:
            # Extract the answer from the response
            answer = response.split("Answer: ")[1]
            return ("answer", answer)
        else:
            # Parse the action from the response
            action = response.split("Action: ")[1]
            if action.startswith("summary_level"):
                # Extract the index from the action
                index = int(action.split("_")[-1])
                return ("action", index)
            elif action == "revert to parent node":
                return ("action", -1)
            else:
                raise ValueError(f"Invalid action: {action}")


# Creating a new MemoryTree instance and saving the tree to memory_tree.json
if not os.path.exists("memory_tree.json"):
    with open(
        "feynman_lectures_chapter_1.txt", "r", encoding="utf-8"
    ) as f:  # Open the file
        text = f.read()  # Read the text from the file
    tree = MemoryTree(text)  # Create a new MemoryTree instance
    print(tree.tree.summary)  # Print the summary of the tree

# Loading the tree from memory_tree.json
with open("memory_tree.json", "r") as f:  # Open the file
    tree_dict = json.load(f)  # Load the tree from the file

# Creating a new Navigation instance and navigating the tree
nav = Navigation(Node(**tree_dict))  # Create a new Navigation instance
print(nav.navigate("What is physics?"))  # Navigate the tree and print the result
