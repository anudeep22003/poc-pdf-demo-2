from langchain.llms import OpenAI
from agents import Agent, classification_agent
from indexer import BuildRagIndex, index_to_product_mapping, product_descriptions

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import json

# import logging

# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
# fh = logging.FileHandler("query.log", mode="a")
# fh.setLevel(logging.INFO)
# logger.addHandler(fh)

###### Pydantic base classes for FastAPI ######


class Message(BaseModel):
    content: str


class Response(BaseModel):
    content: str
    sources: str | None


####################################################


app = FastAPI()

origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)


def perform_rag_call(
    message: Message, index_id: str = "eXP_Family_of_Glass_Mat_Products.pdf"
) -> Response:
    # response query initialize
    response_query = []

    b = BuildRagIndex(index_id)
    response_text, page_numbers = b.query(message.content)
    # sort page numbers for presentation
    page_numbers = sorted(page_numbers)
    response_query.append(response_text)
    response_obj = {
        "content": "\n\n".join(response_query),
        "sources": ", ".join([str(page_num) for page_num in page_numbers]),
    }
    # logger.info(response_obj)
    # logger.info(f"\n {'-'*30}\n")
    return Response(**response_obj)


@app.post("/converse/")
def get_response(
    message: Message,
) -> Response | dict:
    # logger.info(message.content)
    return perform_rag_call(message)


class ConversationHandler:
    def __init__(self, message: Message):
        self.memory: Message | None = None


if __name__ == "__main__":
    # memory_refresher()
    # import uvicorn

    # uvicorn.run(app, port=8001)

    # for doc_location, start_skip, end_skip in documents_to_index:
    #     BuildRagIndex(doc_location, start_skip, end_skip)
    while True:
        query = input("Enter query: ")
        message = Message(content=query)
        print(get_response(message))
