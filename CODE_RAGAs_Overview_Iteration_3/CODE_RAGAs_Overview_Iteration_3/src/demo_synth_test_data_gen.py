import os
import json
import openai
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context

# Open and read the OPENAI_API_KEY 
with open('configs\secrets\secrets.json', 'r') as file:
    configs = json.load(file)
os.environ["OPENAI_API_KEY"] = configs['OPENAI_API_KEY']
openai.api_key = os.environ['OPENAI_API_KEY']

# Loading the sample data using the TextLoader
pages = []
from langchain_community.document_loaders import TextLoader

loader = TextLoader("data\sample.txt", encoding = 'UTF-8')
# load and split the file into pages
local_pages = loader.load_and_split() 
pages.extend(local_pages)

# generator with openai models
model_name = "gpt-3.5-turbo"

# set up LLM and embeddings models
generator_llm = ChatOpenAI(model=model_name)
critic_llm = ChatOpenAI(model=model_name)
embeddings = OpenAIEmbeddings()

# create ragas TestsetGenerator class object
generator = TestsetGenerator.from_langchain(
    generator_llm,
    critic_llm,
    embeddings
)

# Set the resulting question type test set evolutions / distributions-  this should sum to 1.
distributions = {
    simple: 0.3,
    multi_context: 0.2,
    reasoning: 0.5
}

try:    
    test_size = 5 # this indicates the number of question answer pairs to be generated
    
    # generating test sets using ragas
    testset = generator.generate_with_langchain_docs(pages, test_size, distributions) 

    # get results in pandas dataframe
    test_df = testset.to_pandas()

    #  Export results in CSV / Excel (optional)
    timestamp =datetime.now()
    csv_file_name = ".\export\synthetic_data_test_set"+timestamp.strftime("_%m_%d_%Y_ %H_%M_%S")+".csv"
    print(csv_file_name)
    # get results in csv
    test_df.to_csv(csv_file_name, index=False)
except Exception as e:
    print("Error: {}".format(str(e)))

#  command to run the code
#  python .\src\demo_synth_test_data_gen.py