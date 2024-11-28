import os
import json
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.schema import Document

from langchain_core.prompts import PromptTemplate

from output_parsers import feedback_parser, RoleMatch

# Load environment variables from .env file
load_dotenv()

# Streamlit app configuration
st.set_page_config("CV Scanner MVP")
st.header("CV Scanner MVP")

# Initialize global embeddings model
if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

# Preinitialize the LLM
if "llm" not in st.session_state:
    st.session_state.llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

# Upload a PDF file
pdf_file = st.file_uploader(
    "Upload your CV", type="pdf", on_change=st.cache_resource.clear
)


@st.cache_resource
def create_document(pdf):
    pdf_reader = PdfReader(pdf)
    text = "".join(page.extract_text() for page in pdf_reader.pages)

    # Filter content: remove blank lines
    text = "\n".join(line for line in text.split("\n") if line.strip())

    # Create a Document instance with the full text
    document = Document(page_content=text, metadata={})
    return [document]

# Load roles from JSON file
with open("roles.json", "r", encoding="utf-8") as file:
    roles = json.load(file)

# If a PDF file is uploaded
if pdf_file:
    documents = create_document(pdf_file)

    if roles:
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

        new_match_prompt = PromptTemplate(
            input_variables=["documents", "roles"],
            partial_variables={
                "format_instructions": feedback_parser.get_format_instructions
            },
            template="""
                Based on the provided CV information: {documents}
                and the following roles: {roles},

                Please perform the following tasks:

                1. **List the Candidate's Main Skills and Experiences**:
                - Identify and enumerate the 5 primary skills and experiences mentioned in the CV.

                2. **Role Match**:
                - For each role, provide:
                    - The role name.
                    - A list of skills from the candidate's CV (not the main skills) that fit the role (5 maximum).
                    - If there are no relevant skills, indicate with a single bullet:
                        * There are no skills that fit the job position.
                    - A match score from 0 to 100 indicating how well the candidate fits the role.

                \n{format_instructions}
            """,
        )

        chain = new_match_prompt | st.session_state.llm | feedback_parser
        res = chain.invoke(input={"documents": documents, "roles": roles})

        st.write("## Main Skills of the candidate")
        st.markdown("\n".join([f"- {skill}" for skill in res.main_skills]))


        # Dios haz que acabe esto ya
        st.write("## Role Match")
        for role_match in res.role_matches:
            st.write(f"#### Name: {role_match.rol_name}")
            st.write("Skills that fit the role:")
            if role_match.fit_skills:
                st.markdown(
                    "\n".join([f"- {skill}" for skill in role_match.fit_skills])
                )
            else:
                st.markdown("- There are no skills that fit the job position")
            st.write(f"**Match Score: {role_match.match_score}%**")

        # Demonstration purposes:
        # st.write("### Response:")
        # st.write(res)