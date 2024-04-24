from datetime import datetime
from src.llm.assistant import Turbo4
from src.types import Chat, TurboTool
from typing import List, Callable
import os
import uuid
from src.agent.instruments import PostgresAgentInstruments
from src.utils import llm
from src.db import db_embeddings as embeddings
import argparse

DB_URL = os.environ.get("DATABASE_URL")
POSTGRES_TABLE_DEFINITIONS_CAP_REF = "TABLE_DEFINITIONS"


custom_function_tool_config = {
    "type": "function",
    "function": {
        "name": "store_fact",
        "description": "A function that stores a fact.",
        "parameters": {
            "type": "object",
            "properties": {"fact": {"type": "string"}},
        },
    },
}

run_sql_tool_config = {
    "type": "function",
    "function": {
        "name": "run_sql",
        "description": "Run a SQL query against the postgres database",
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "The SQL query to run",
                }
            },
            "required": ["sql"],
        },
    },
}


def generate_session_id(raw_prompt: str):
    """
    "get jobs with 'Completed' or 'Started' status"

    ->

    "get_jobs_with_Completed_or_Started_status__12_22_22"
    """

    now = datetime.now()
    hours = now.hour
    minutes = now.minute
    seconds = now.second

    short_time_mm_ss = f"{hours:02}_{minutes:02}_{seconds:02}"

    lower_case = raw_prompt.lower()
    no_spaces = lower_case.replace(" ", "_")
    no_quotes = no_spaces.replace("'", "")
    shorter = no_quotes[:30]
    with_uuid = shorter + "__" + short_time_mm_ss
    return with_uuid


def store_fact(fact: str):
    print(f"------store_fact({fact})------")
    return "Fact stored."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", help="The prompt for the AI")
    args = parser.parse_args()

    if not args.prompt:
        print("Please provide a prompt")
        return

    raw_prompt = args.prompt

    prompt = f"Fulfill this database query: {raw_prompt}. "

    assistant_name = "Turbo4"

    assistant = Turbo4()

    session_id = generate_session_id(assistant_name + raw_prompt)

    with PostgresAgentInstruments(DB_URL, session_id) as (agent_instruments, db):
        database_embedder = embeddings.DatabaseEmbedder(db)

        table_definitions = database_embedder.get_similar_table_defs_for_prompt(
            raw_prompt
        )

        prompt = llm.add_cap_ref(
            prompt,
            f"Use these {POSTGRES_TABLE_DEFINITIONS_CAP_REF} to satisfy the database query.",
            POSTGRES_TABLE_DEFINITIONS_CAP_REF,
            table_definitions,
        )

        tools = [
            TurboTool("run_sql", run_sql_tool_config, agent_instruments.run_sql),
        ]

        (
            assistant.get_or_create_assistant(assistant_name)
            .set_instructions(
                "You're an elite SQL developer. You generate the most concise and performant SQL queries."
            )
            .equip_tools(tools)
            .make_thread()
            .add_message(prompt)
            .run_thread()
            .add_message(
                "Use the run_sql function to run the SQL you've just generated.",
            )
            .run_thread(toolbox=[tools[0].name])
            .run_validation(agent_instruments.validate_run_sql)
            .spy_on_assistant(agent_instruments.make_agent_chat_file(assistant_name))
            .get_costs_and_tokens(
                agent_instruments.make_agent_cost_file(assistant_name)
            )
        )

        print(f"✅ Turbo4 Assistant finished.")

        # ---------- Simple Prompt Solution - Same thing, only 2 api calls instead of 8+ ------------
        # sql_response = llm.prompt(
        #     prompt,
        #     model="gpt-4-1106-preview",
        #     instructions="You're an elite SQL developer. You generate the most concise and performant SQL queries.",
        # )
        # llm.prompt_func(
        #     "Use the run_sql function to run the SQL you've just generated: "
        #     + sql_response,
        #     model="gpt-4-1106-preview",
        #     instructions="You're an elite SQL developer. You generate the most concise and performant SQL queries.",
        #     turbo_tools=tools,
        # )
        # agent_instruments.validate_run_sql()

        # ----------- Example use case of Turbo4 and the Assistants API ------------

        # (
        #     assistant.get_or_create_assistant(assistant_name)
        #     .make_thread()
        #     .equip_tools(tools)
        #     .add_message("Generate 10 random facts about LLM technology.")
        #     .run_thread()
        #     .add_message("Use the store_fact function to 1 fact.")
        #     .run_thread(toolbox=["store_fact"])
        # )


if __name__ == "__main__":
    main()