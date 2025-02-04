from datetime import datetime

import requests
import os
import yaml
import json
import re
import time
import pandas as pd
import torch

from typing import Iterator
from pathlib import Path
from openai import OpenAI

from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document as LCDocument

from langchain_openai import ChatOpenAI
from langchain_community.llms import VLLMOpenAI
from langchain_milvus import Milvus
from langchain_text_splitters import RecursiveCharacterTextSplitter

from docling.document_converter import DocumentConverter

MILVUS_URI = os.getenv("MILVUS_URI", "./milvus_custom_eval.db")
MILVUS_USERNAME = os.getenv("MILVUS_USERNAME", "")
MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD", "")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "my_org_documents")

MAX_TOKENS = 2048
TEMPERATURE = 0.00
ILAB_TUNED_MODEL_PROMPT = """<|system|>
I am a Red Hat Instruct Model, an AI language model developed by Red Hat and IBM Research based on the granite-3.0-8b-base model.
My primary role is to serve as a chat assistant.
<|user|>
Answer the following question based on internal knowledge.
Question: {question}
Answer:
<|assistant|>
"""

ILAB_TUNED_MODEL_RAG_PROMPT = """<|system|>
I am a Red Hat Instruct Model, an AI language model developed by Red Hat and IBM Research based on the granite-3.0-8b-base model.
My primary role is to serve as a chat assistant.
<|user|>
Context:
{context}
Answer the following question from the above context and internal memory.
Question: {question}
Answer:
<|assistant|>
"""

class DoclingPDFLoader(BaseLoader):
    def __init__(self, file_path: str | list[str]) -> None:
        self._file_paths = file_path if isinstance(file_path, list) else [file_path]
        self._converter = DocumentConverter()

    def lazy_load(self) -> Iterator[LCDocument]:
        for source in self._file_paths:
            dl_doc = self._converter.convert(source).document
            text = dl_doc.export_to_markdown()
            yield LCDocument(page_content=text)

def replace_special_char(original_str):
    return re.sub(r"[^\w]", "_", original_str)


def write_jsonl(jsonl_file_path, data: list[dict]):
    with open(jsonl_file_path, 'w') as jsonl_file:
        for entry in data:
            jsonl_file.write(json.dumps(entry) + "\n")


def read_jsonl(jsonl_file_path) -> list[dict]:
    data = []
    with open(jsonl_file_path, 'r') as jsonl_file:
        for line in jsonl_file:
            data.append(json.loads(line.strip()))
    return data


def get_config():
    with open("llm_config.yaml", "r") as f:
        llm_config = yaml.safe_load(f)
    return llm_config


def get_first_config(RAG=False):
    llm_config = get_config()
    for testing_config in llm_config["testing_config"]:
        if (RAG and testing_config.get("rag")) or ((not RAG) and (not testing_config.get("rag"))):
            return testing_config
        else:
            continue


def get_output_dir(parent=""):
    llm_config = get_config()
    output_directory = replace_special_char(llm_config.get("name", "output"))
    if parent:
        output_directory = f"{parent}/{output_directory}"
    os.makedirs(output_directory, exist_ok=True)
    return output_directory


def get_testing_config_name(testing_config):
    name = testing_config.get("name")
    if name:
        return replace_special_char(name)

    name = testing_config.get("model_name")
    if name and testing_config.get("rag"):
        name = replace_special_char(name + "_rag")
    return name


def create_llm(endpoint_url, model_name, api_key, model_type="vllm"):
    openai_api_key = re.sub(r"\s+", "", api_key)
    if model_type == "openai":
        return ChatOpenAI(
            openai_api_key=openai_api_key,
            model_name=model_name,
            streaming=False)
    return VLLMOpenAI(
        openai_api_key=openai_api_key,
        openai_api_base=endpoint_url,  #https://model...com/v1
        model_name=model_name,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        streaming=False
    )


def chat_request(llm, template_str, question, context=None):
    num_retries = 1
    for attempt in range(num_retries):
        try:
            prompt = PromptTemplate.from_template(template_str)
            chain = prompt | llm | StrOutputParser()
            params = {"question": question}
            if context:
                params["context"] = context
            response = chain.invoke(params)
            return response.strip()
        except Exception as e:
            print(f"Request failed: {e}")
            if attempt < num_retries:
                print(f"Retrying in 5 seconds...")
                time.sleep(5)
            else:
                return ""

def get_reference_answers(directory: str) -> list[{str: str}]:
    reference_answers_df = pd.DataFrame(columns=["user_input", "reference"])
    for file_path in Path(directory).rglob('*.csv'):
        csv_df = pd.read_csv(file_path)
        print(f"{file_path}: {csv_df.shape[0]} questions")
        reference_answers_df = pd.concat([reference_answers_df, csv_df], ignore_index=True)


    for file_path in Path(directory).rglob('*.jsonl'):
        jsonl_df = pd.read_json(file_path, orient="records", lines=True)
        print(f"{file_path}: {jsonl_df.shape[0]} questions")
        reference_answers_df = pd.concat([reference_answers_df, jsonl_df], ignore_index=True)

    qna_list = []

    for file_path in Path(directory).rglob('*.yaml'):
        with open(file_path) as file:
            qna = yaml.load(file, Loader=yaml.FullLoader)
            for seed_example in qna["seed_examples"]:
                for questions_and_answers in seed_example["questions_and_answers"]:
                    qna_list.append({
                        "user_input": questions_and_answers["question"].strip(),
                        "reference": questions_and_answers["answer"].strip()
                    })
            print(f"{file_path}: {len(qna_list)} questions")

    reference_answers_df = pd.concat([reference_answers_df, pd.DataFrame(qna_list)], ignore_index=True)
    reference_answers_df = reference_answers_df.drop_duplicates(subset=["user_input"])
    return reference_answers_df.to_dict(orient="records")


def create_document_db(
        milvus_uri: str="milvus_eval.db",
        milvus_username: str= "",
        milvus_password: str="",
        milvus_collection: str="my_org_documents",
        drop_old: bool=True
) -> Milvus:
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    model_kwargs = {"trust_remote_code": True, "device": device}
    embeddings = HuggingFaceEmbeddings(
        model_name="nomic-ai/nomic-embed-text-v1.5",
        model_kwargs=model_kwargs,
        show_progress=True
    )

    db = Milvus(
        embedding_function=embeddings,
        connection_args={
            "uri": milvus_uri,
            "user": milvus_username,
            "password": milvus_password
        },
        collection_name=milvus_collection,
        auto_id=True,
        drop_old=drop_old
    )
    return db


def load_docs(db: Milvus, document_directory: str):
    file_paths = [str(path) for path in Path(document_directory).rglob('*.pdf')]
    loader = DoclingPDFLoader(file_path=file_paths)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=100,
    )

    docs = loader.load()
    splits = text_splitter.split_documents(docs)
    loaded = db.add_documents(splits)
    print(f"{len(loaded)} documents loaded.")
    return loaded


def get_context(reference_answers: list[dict], document_directory: str) -> list[dict]:
    generated_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    db_filename = f"./{generated_str}.db"
    db_lockfile = f"./.{generated_str}.db.lock"
    db = create_document_db(milvus_uri=db_filename)
    load_docs(db, document_directory)

    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    reference_answers_with_context = []
    for item in reference_answers:
        new_item = item.copy()
        docs = retriever.invoke(new_item["user_input"])
        retrieved_context = "\n\n".join(doc.page_content for doc in docs)
        new_item["retrieved_context"] = retrieved_context
        reference_answers_with_context.append(new_item)

    os.remove(db_filename)
    os.remove(db_lockfile)
    return reference_answers_with_context


def summarize_results(results: list[dict]) -> pd.DataFrame:
    summary_output_df = pd.DataFrame()
    for result in results:
        result_name = result.get("name")
        summary_output_df[f"{result_name}"] = [score.get("score") for score in result["scores"] ]

    sum_row = summary_output_df.sum(axis=0, numeric_only=True)
    average_row = summary_output_df.mean(axis=0, numeric_only=True)

    question_indices = [f"Q{i + 1}" for i in range(len(summary_output_df))]
    question_indices.append("Sum")
    question_indices.append("Average")
    summary_output_df.loc[len(summary_output_df)] = sum_row
    summary_output_df.loc[len(summary_output_df)] = average_row
    summary_output_df.insert(0, 'question index', question_indices)
    return summary_output_df

def write_excel(summary_df, results, output_file):
    with pd.ExcelWriter(output_file) as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

        for result in results:
            result_name = result.get("name")
            scores_df = pd.DataFrame(result.get("scores"))
            scores_df.to_excel(writer, sheet_name=f"{result_name}"[:30])

def write_markdown(summary_df, results, output_file):
    with open(output_file, 'w') as f:
        f.write(f"## Summary\n")
        f.write(summary_df.to_markdown(index=False))
        f.write("\n\n")
        for result in results:
            result_name = result.get("name")
            scores_df = pd.DataFrame(result.get("scores"))
            f.write(f"## {result_name}\n")
            f.write(scores_df.to_markdown(index=False))
            f.write("\n\n")

def write_html(summary_df, results, output_file):
    with open(output_file, 'w') as f:
        f.write(f"<h2>Summary</h2>\n")
        f.write(summary_df.to_html(index=False))
        f.write("\n\n")
        for result in results:
            result_name = result.get("name")
            scores_df = pd.DataFrame(result.get("scores"))
            f.write(f"<h2>{result_name}</h2>\n")
            f.write(scores_df.to_html(index=False))
            f.write("\n\n")
