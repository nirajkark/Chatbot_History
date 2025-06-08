import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from dotenv import load_dotenv
from langchain.callbacks import StreamlitCallbackHandler
st.set_page_config(page_title="Text to Math Problem Solver And  Data Search Assistent",page_icon="@")
st.title("Text to Math Problem Solver")
groq_api_key=st.sidebar.text_input(label="Groq Api key",type="password")
if groq_api_key:
    st.info("please add your Groq API to continue")
    st.stop()
llm=ChatGroq(model="Gemma2-9b-It",groq_api_key=groq_api_key)
wikipedia_wrapper=WikipediaAPIWrapper()
wikipedia_tool=Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="A tool for searching the internet and solving your math problem"
)
#initiliazing the math tool