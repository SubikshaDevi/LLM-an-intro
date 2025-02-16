import requests
from dotenv import load_dotenv
import os
 
env = '.env'
load_dotenv(env) 
hugging_face_api = os.environ.get('hugging_face_api')
print(hugging_face_api)

def test():
    API_URL = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment-latest"
    headers = {"Authorization": f"Bearer {hugging_face_api}"}
    payload = {
        "inputs": "Today is a great day",
    }

    response = requests.post(API_URL, headers=headers, json=payload)
    print(response.json())


def chat_test():
    from langchain_huggingface import HuggingFaceEndpoint
    from langchain.chains import LLMChain
    from langchain_core.prompts import PromptTemplate

    # Question and context for the model
    # question = "Who won the FIFA World Cup in the year 1994?"
    # context = "The FIFA World Cup is an international football tournament held every four years. The 1994 FIFA World Cup was held in the United States and was won by Brazil."

    # # Define the prompt template
    # template = """Question: {question}
    # Context: {context}

    # Answer: Let's think step by step."""

    # prompt = PromptTemplate.from_template(template)

    # # repo_id = "google-bert/bert-large-uncased-whole-word-masking-finetuned-squad"
    # repo_id  = "deepset/roberta-base-squad2"

    # # Initialize the HuggingFaceEndpoint with the correct task
    # llm = HuggingFaceEndpoint(
    #     repo_id=repo_id,
    #     task="question-answering",  # Set the correct task type
    #     max_length=128,
    #     temperature=0.5,
    #     huggingfacehub_api_token=hugging_face_api,
    # )

    # # Use the correct input format for question-answering models
    # llm_chain = prompt | llm
    # response = llm_chain.invoke({"question": question, "context": context})  # Pass both question & context
    # print(response)


    from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
    import getpass
    import os

    os.environ["hugging_face_api"] = getpass.getpass(
        "Enter your Hugging Face API key: "
    )

    llm = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        task="text-generation",
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
    )

    chat_model = ChatHuggingFace(llm=llm)

    from langchain_core.messages import (
        HumanMessage,
        SystemMessage,
    )

    messages = [
        SystemMessage(content="You're a helpful assistant"),
        HumanMessage(
            content="Who won the FIFA World Cup in the year 1994?"
        ),
    ]

    ai_msg = chat_model.invoke(messages)
    print(ai_msg.content)
    
    
def youtube():
    from langchain.document_loaders import YoutubeLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain.vectorstores import FAISS
    from langchain.llms import OpenAI
    from langchain import PromptTemplate
    from langchain.chains import LLMChain
    from dotenv import find_dotenv, load_dotenv
    import textwrap
    from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

    load_dotenv(env)
    embeddings = HuggingFaceEmbeddings()


    def create_db_from_youtube_video_url(video_url: str) -> FAISS:
        loader = YoutubeLoader.from_youtube_url(video_url)
        transcript = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(transcript)

        db = FAISS.from_documents(docs, embeddings)
        return db
    
    def get_response_from_query(db, query, k=4):
        """
        text-davinci-003 can handle up to 4097 tokens. Setting the chunksize to 1000 and k to 4 maximizes
        the number of tokens to analyze.
        """

        docs = db.similarity_search(query, k=k)
        docs_page_content = " ".join([d.page_content for d in docs])
        from langchain_huggingface.llms import HuggingFacePipeline

        llm = HuggingFacePipeline.from_model_id(
            model_id="gpt2",
            task="text-generation",
            pipeline_kwargs={"max_new_tokens": 10},
        )
        # llm = OpenAI(model_name="text-davinci-003")

        prompt = PromptTemplate(
            input_variables=["question", "docs"],
            template="""
            You are a helpful assistant that that can answer questions about youtube videos 
            based on the video's transcript.
            
            Answer the following question: {question}
            By searching the following video transcript: {docs}
            
            Only use the factual information from the transcript to answer the question.
            
            If you feel like you don't have enough information to answer the question, say "I don't know".
            
            Your answers should be verbose and detailed.
            """,
        )

        chain = LLMChain(llm=llm, prompt=prompt)

        response = chain.run(question=query, docs=docs_page_content)
        response = response.replace("\n", "")
        return response, docs

    video_url = "https://www.youtube.com/watch?v=L_Guz73e6fw"
    db = create_db_from_youtube_video_url(video_url)
    print(db)
    query = "What are they saying about Microsoft?"
    response, docs = get_response_from_query(db, query)
    print(textwrap.fill(response, width=85))
    
if __name__ == "__main__":
    youtube()