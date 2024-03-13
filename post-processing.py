"""
This script processes the extracted elements (from pdfs)

@Qingshi Tu
"""


"""
Import libraries
"""
import openai
import langchain_openai
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
import os
import glob
import base64
import pandas as pd


"""
Config
"""
# path to the extracted elements
# same folder to store the post-processed elements
output_folder = "/home/bizon/Documents/code_projects/tiangong_ai_unstructured_dev/output"

vision_client = openai.OpenAI()
llm = langchain_openai.OpenAI(temperature=0)

"""
Define functions
"""
def vision_completion(image_path: str) -> str:
    """
    This function is adapted from TianGong-AI-Unstructure > src > tools > vision.py
    """
    # Open the image file in binary mode
    with open(image_path, 'rb') as image_file:
        # Convert the binary data to base64
        image_data = base64.b64encode(image_file.read()).decode("utf-8")
    
    try:
        response = vision_client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What is in this image? Only return neat facts in English.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}",
                            },
                        },
                    ],
                }
            ],
            max_tokens=300,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error analyzing image: {e}")
        return None


def image_screening(image_path: str) -> str:
    """
    (1) use gpt-4 vision to generate a summary of the image
    (2) then ask it to decide whether the image belongs to one of the following groups:
        - system boundary chart of an life cycle assessment study
        - impact assessment results of an life cycle assessment study
    (3) remove the image if it does not belong any of the abovementioned groups
    """

    # [step 1] use gpt-4 vision to generate a summary of the image
    image_summary = vision_completion(image_path)
    # print(image_summary)

    # [step 2] decide whether or not the image belongs to the expected groups
    # define the template with explicit input variables
    template_string = """
        Given the context and specific details provided, please answer the following question.

        Context: In a life cycle assessment study, there two types of figures:  (1) system boudary diagram: visually delineates the processes and flows to be included in the LCA, marking the limits of the analysis; 
        (2) impact assessment results: visually represents the outcomes of the environmental impacts assessed, illustrating the magnitude and distribution of impacts across different categories.

        Question: based on the content of "{image_summary}", decide whether it is describing a system boundry diagram, an impact assessment results diagram, or neither. Output your answer

        Answer: """

    prompt_template = PromptTemplate(
        template=template_string,
        input_variables=["image_summary"]
    )

    # make the query to decide whether or not the image belongs to the expected groups
    query = prompt_template.template.format(image_summary=image_summary)
    return llm(query)

def text_synthesize(text_path: str) -> str:
    """
    synthesize the key information from the text elements
    """

    # prompt for text synthesis
    prompt = """

    """


def table_analyze(xlsx_file_path: str, sheet_of_interest: str, user_query: str) -> str:
    """
    analyze the content of a table using llm. Supported file type:
    - .xlsx
    """

    ## parse the table into text string
    # load the xlsx file
    xlsx = pd.ExcelFile(xlsx_file_path)
    # Initialize a dictionary to store text representations of each sheet
    sheets_text = {}

    # Loop through each sheet in the xlsx file
    for sheet_name in xlsx.sheet_names:
        # Read the sheet into a DataFrame
        df = pd.read_excel(xlsx, sheet_name=sheet_name)
    
        # Convert the DataFrame to a text string, here we use CSV format as an intermediary
        text_string = df.to_csv(index=False, sep=',')
    
        # Store the text string in our dictionary, using the sheet name as the key
        sheets_text[sheet_name] = text_string

    ## prompt for table analysis
    prompt = f"{sheets_text[sheet_of_interest]}\n\n{user_query}"

    return llm(prompt)

if __name__ == "__main__":

    ## image screening
    # # get a list of image paths
    # pattern = os.path.sep.join([output_folder, '*.jpg'])
    # image_path_list = glob.glob(pattern)
    # # for image_path in image_path_list:
    # #     print(image_path)

    # # loop over the paths and retain only the ones relevant to LCA
    # decision_dict = {}

    # for image_path in image_path_list:
    #     decision_dict[image_path.split("/")[-1]] = image_screening(image_path)

    # # convert dict to df
    # data = list(decision_dict.items())
    # decision_df = pd.DataFrame(data, columns=['ImagePath', 'Decision'])
    # print(decision_df)

    # # save the results in csv to the output folder
    # decision_df.to_csv(os.path.sep.join([output_folder,'image_screening_results.csv']), index=False)

    ## table analysis
    xlsx_file_path = os.path.sep.join([output_folder, 'output_tables.xlsx'])
    sheet_of_interest = 'extracted_tab_3'
    user_query = 'what is the amount of Polyurethane (finger joint), please answer using only the information provided in prompt'

    response = table_analyze(xlsx_file_path,sheet_of_interest,user_query)
    print(response)


