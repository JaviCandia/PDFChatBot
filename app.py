import os

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.schema import Document

from langchain_core.prompts import PromptTemplate

from output_parsers import feedback_parser

# Load environment variables from .env file
load_dotenv()

# Streamlit app configuration
st.set_page_config('CV Scanner MVP')
st.header("CV Scanner MVP")

# Initialize global embeddings model
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Preinitialize the LLM
if 'llm' not in st.session_state:
    st.session_state.llm = ChatOpenAI(model_name='gpt-4o-mini', temperature=0)

# Upload a PDF file
pdf_file = st.file_uploader("Upload your CV", type="pdf", on_change=st.cache_resource.clear)

@st.cache_resource
def create_document(pdf):
    pdf_reader = PdfReader(pdf)
    text = "".join(page.extract_text() for page in pdf_reader.pages)

    # Filter content: remove blank lines
    text = "\n".join(line for line in text.split("\n") if line.strip())

    # Create a Document instance with the full text
    document = Document(page_content=text, metadata={})
    return [document]

# If a PDF file is uploaded
if pdf_file:
    documents = create_document(pdf_file)
    job_description = st.text_area("Provide the job description:")

    if job_description:
        os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

        new_match_prompt = PromptTemplate(
            input_variables=["documents", "job_description"],
            partial_variables={"format_instructions": feedback_parser.get_format_instructions},
            template="""
                Based on the provided CV information: {documents}
                and the job position details: {job_description},

                Please perform the following tasks:

                1. **List the Candidate's Main Skills and Experiences**:
                - Identify and enumerate the 5 primary skills and experiences mentioned in the CV.

                2. **Candidate Skills that fit Job Position**:
                - Provide a list of 5 skills from the candidate's CV that directly fit the job position.
                - If there are no relevant skills, indicate with a single bullet:
                    * There are no skills that fit the job position.

                3. **Match Score: -right here provide a match score from 0 to 100 indicating how well the candidate fits the job position-**
                - Use 0 if the job position has nothing to do with the candidate's skills or is in a completely different field.
                - Justify the match score by highlighting key similarities or discrepancies between the candidate's qualifications and the job requirements.
                \n{format_instructions}
            """
        )

        chain = new_match_prompt | st.session_state.llm | feedback_parser
        res = chain.invoke(input = {"documents": documents, "job_description": job_description})

        st.write("### Mains Skills of the candidate")
        st.markdown("\n".join([f"- {skill}" for skill in res.main_skills]))

        st.write("### Skills that fit the job position")
        if res.fit_skills:
            st.markdown("\n".join([f"- {skill}" for skill in res.fit_skills]))
        else:
            st.markdown("- There are no skills that fit the job position")

        st.write(f"### Match Score: {res.match_score}%")

        # Demonstration purposes:
        # st.write("### Response:")
        # st.write(res)