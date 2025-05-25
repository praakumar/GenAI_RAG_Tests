import os
import json

# set up openai api key
f = open(r'configs\secrets\secrets.json')
data = json.load(f)
os.environ["OPENAI_API_KEY"] = data['OPENAI_API_KEY']

# import RAGAS packages
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

# Note: We are mocking the response of RAG pipeline below in a Python dictionary for demo purposes.
data_samples = {
    'question': ['Where and when was Einstein born?',
                 'Where and when was Einstein born?',
                 'When was the first super bowl?',
                 'Who won the most super bowls?',
                 'Who won the T20 cricket world cup in 2024?'],

# ground_truth: This is human annotated information here but this can be automated - will understand with implementation shortly
    'ground_truth': ['Einstein was born in Germany on 14th March 1879.',
                     'Einstein was born in Germany on 14th March 1879.',
                     'The first superbowl was held on January 15, 1967',
                     'The New England Patriots have won the Super Bowl a record six times',
                     'India won the T20 cricket world cup in 2024'],

    'answer': ['Einstein was born in Germany on 14th March 1879.',
               'Einstein was born in Germany on 20th March 1879.',
               'The first superbowl was held on Jan 15, 1967',
               'The most super bowls have been won by The New England Patriots',
               'India won the T20 cricket world cup in 2024'],

    'contexts': [
        ['Albert Einstein (born 14 March 1879) was a German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time'],
        ['Albert Einstein (born 14 March 1879) was a German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time'],
        ['The First AFL–NFL World Championship Game was an American football game played on January 15 1967, at the Los Angeles Memorial Coliseum in Los Angeles,'],
        ['The Green Bay Packers...Green Bay, Wisconsin.', 'The Packers compete...Football Conference'],
        ['India won the T20 cricket world cup in 2024']
    ]

}

# convert dictionary inro Dataset
dataset = Dataset.from_dict(data_samples)

# create list of imported ragas metrics
metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

# call the ragas evaluate function
score = evaluate(dataset, metrics=metrics)

# get scores in dataframe and export it in csv or excel (optional)
df = score.to_pandas()
print(df)
df.to_csv(r'.\export\1_intro_score.csv', index=False)

#  command to run the code
#  python .\src\ragas_intro.py