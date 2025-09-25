import os
import ast
import re
import streamlit as st
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain.schema import Document
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.agents.agent_toolkits import create_retriever_tool
from langgraph.prebuilt import create_react_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')
if openai_api_key is None:
    st.error("OPENAI_API_KEY is not set")
else:
    st.success("OPENAI_API_KEY is set")


connection_string = os.getenv('DATABASE_URL')


def get_connection():
    try:
        db = SQLDatabase.from_uri(connection_string)
        return db
    except Exception as e:
        st.error(f'Connection with DB failed {e}')

def get_tables():
    try:
        db = get_connection()
        return db.get_usable_table_names()
    except Exception as e:
        st.error(f"Error getting tables: {e}")
        return []
    

def query_as_list(db, query):
            res = db.run(query)
            res = [el for sub in ast.literal_eval(res) for el in sub if el]
            res = [re.sub(r"\b\d+\b", "", string).strip() for string in res]
            return list(set(res))    


def get_metadata(table_name):
    try:
        db = get_connection()
        query = f"""
                SELECT 
                    column_name, 
                    data_type, 
                    is_nullable,
                    column_default,
                    (SELECT pg_catalog.col_description(c.oid, cols.ordinal_position::int)
                    FROM pg_catalog.pg_class c
                    WHERE c.oid = (SELECT ('"' || cols.table_name || '"')::regclass::oid)
                        AND c.relname = cols.table_name) AS column_comment
                FROM information_schema.columns cols
                WHERE table_name = '{table_name}'
        """
        result_str = db.run(query)
        columns = ast.literal_eval(result_str)
        metadata = [
            f"Field: {col[0]}, Type: {col[1]}, Null: {'YES' if col[2]=='YES' else 'NO'}, "
            
            f"Default: {col[3]}, Comment: {col[4] or ''}"
            for col in columns
        ]
        return metadata
    except Exception as e:
        st.error(f"Error getting metadata: {e}")
        return []


def get_pronouns(table_name):
    try:
        db = get_connection()
        query = f"""
                SELECT 
                    column_name, 
                    data_type, 
                    is_nullable,
                    column_default,
                    (SELECT pg_catalog.col_description(c.oid, cols.ordinal_position::int)
                    FROM pg_catalog.pg_class c
                    WHERE c.oid = (SELECT ('"' || cols.table_name || '"')::regclass::oid)
                        AND c.relname = cols.table_name) AS column_comment
                FROM information_schema.columns cols
                WHERE table_name = '{table_name}'
        """
        result_str = db.run(query)
        columns = ast.literal_eval(result_str)
        arr_pronouns = []
        for col in columns:
            arr_pronouns += query_as_list(db, f"SELECT {col[0]} FROM iron_ore_blocks")

    except Exception as e:
        st.error(f"Error getting metadata: {e}")
        return []
    finally:
        db._engine.dispose()


def get_metadata_tool():

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key = openai_api_key
    )

    tables = get_tables()
    table_metadata = {}
    for table in tables:
         table_metadata[table] = get_metadata(table)

    document = []
    for table_name, fields in table_metadata.items():
         table_info = f"Table: {table_name}\n"
         table_info += '\n'.join(fields)
         document.append(Document(page_content=table_info, metadata={"source": table_name}))
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(document)
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    description = (
        "Use to get idea about the table"
        "this has table names and every column in the table along with table comments"
        "column datatype etc.."
    )

    retriever_tool = create_retriever_tool(
        retriever,
        name="search_proper_nouns",
        description=description,
    )
    
    return retriever_tool

def load_data():
    llm = ChatOpenAI(
        model = 'gpt-5-mini-2025-08-07',
        api_key = openai_api_key
    )

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key = openai_api_key
    )

    retriever_tool = get_metadata_tool()

    # Add to system message
    suffix = (
        "If you need to filter on a proper noun like a Name, you must ALWAYS first look up "
        "the filter value using the 'search_proper_nouns' tool! Do not try to "
        "guess at the proper name - use this function to find similar ones."
    )

    system_message = """
    You are an agent designed to interact with a SQL database.
    Given an input question, create a syntactically correct {dialect} query to run,
    then look at the results of the query and return the answer. Unless the user
    specifies a specific number of examples they wish to obtain, always limit your
    query to at most {top_k} results.

    You can order the results by a relevant column to return the most interesting
    examples in the database. Never query for all the columns from a specific table,
    only ask for the relevant columns given the question.

    You MUST double check your query before executing it. If you get an error while
    executing a query, rewrite the query and try again.

    DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
    database.

    To start you should ALWAYS look at the tables in the database to see what you
    can query. Do NOT skip this step.

    Then you should query the schema of the most relevant tables.
    """.format(
        dialect=db.dia,
        top_k=5,
    )

    system = f"{system_message}\n\n{suffix}"

    db = get_connection()

    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    tools = toolkit.get_tools()

    tools.append(retriever_tool)

    agent = create_react_agent(llm, tools, prompt=system)

    return agent

def main():
    st.title("Chat on Iron ore blocks")
    agent = load_data()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if question := st.chat_input("What would you like to know about the database?"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            result = agent.invoke({"messages": [{"role": "user", "content": question}]})

        st.markdown(result["messages"][-1].content)

    
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["messages"][-1].content
        })
    
    st.sidebar.header("Sample Queries")

    sample_queries = [
        "how many blocks are there in odisha",
        "Which block was won at highest premium and give me all the details of the specific block",
        "In the last financial year how many blocks where auctioned also explain what you are taking as financial year",
        "How many blocks LOI (with extension) is about to expire in the current financial year"


    ]

    for i, query in enumerate(sample_queries):
        st.sidebar.subheader(f"Sample Query {i+1}")
        st.sidebar.text(query)



if __name__ == "__main__":
    main()