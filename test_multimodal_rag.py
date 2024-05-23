from unstructured.chunking.title import chunk_by_title
from unstructured.cleaners.core import (
    clean,
    group_broken_paragraphs,
)
from unstructured.documents.elements import Footer, Header, Image, CompositeElement, Table
from unstructured.partition.auto import partition
from unstructured.partition.image import partition_image
from unstructured.partition.pdf import partition_pdf
from tools.vision import vision_completion
import pandas as pd
from typing import List
from io import StringIO
import os
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
import weaviate
from weaviate.classes.config import Configure, DataType, Property


"""
====
Config information
====
"""
input_folder = "/home/bizon/Documents/code_projects/tiangong_ai_unstructured_dev/input"
output_folder = "/home/bizon/Documents/code_projects/tiangong_ai_unstructured_dev/output"
local_db_folder = "/home/bizon/Documents/code_projects/tiangong_ai_unstructured_dev/local_db"
pdf_name = os.path.sep.join([input_folder,"CtoG-LCA-Canadian-CLT.pdf"])
image_input_folder = os.path.sep.join([input_folder,"image_input"])


"""
====
Code for multimodal query
====
"""
# get elements from a pdf
elements = partition_pdf(pdf_name)

# create a client
MM_EMBEDDING_API_KEY = os.getenv('GOOGLE_AI_API_KEY')
TEXT_EMBEDDING_API_KEY = os.getenv('OPENAI_API_KEY')

client = weaviate.connect_to_embedded(
    version="1.24.4",
    environment_variables={
        "ENABLE_MODULES": "multi2vec-palm, text2vec-openai"
    },
    headers={
        "X-PALM-Api-Key": MM_EMBEDDING_API_KEY,
        "X-OpenAI-Api-Key": TEXT_EMBEDDING_API_KEY
    }
)

client.is_ready()

# create a collection
client.collections.create(
    name="TestLCIPdfs",
    properties=[
        Property(name='title', data_type=DataType.TEXT),
        Property(name='abstract', data_type=DataType.TEXT),
        Property(name='image', data_type=DataType.BLOB),
    ],

    # define and configure the vector spaces
    vectorizer_config=[
        # vectorize the pdf title & abstract
        Configure.NamedVectors.text2vec_openai(
            name='txt_vector',
            source_properties=['title','abstract'],
        ),

        # vectorize the images
        Configure.NamedVectors.multi2vec_palm(
            name="img_vector",
            image_fields=['image'],

            project_id="semi-random-dev",
            location="us-central1",
            model_id="multimodalembedding@001",
            dimensions=1408,
        ),
    ]

)

