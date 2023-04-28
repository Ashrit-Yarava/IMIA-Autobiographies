import pandas as pd
from pipeline import QuestionAnswering
import json

import warnings
warnings.filterwarnings("ignore")


question = "What did {} study in school?"

# Load the model and set up the pipeline.
model, tokenizer = QuestionAnswering.load_model()
qa = QuestionAnswering(model, tokenizer)


# Load the biographies.
americas_bios = json.loads(open("../data/jsons/americas.json").read())
europe_bios = json.loads(open("../data/jsons/europe.json").read())
africa_asia_bios = json.loads(open("../data/jsons/africaasia.json").read())

bios = {**americas_bios, **europe_bios, **africa_asia_bios}


author_answers = []


authors_to_question = list(bios.keys())
for i, author in enumerate(authors_to_question):
    bio = f"I am {author}. " + bios[author]

    response = qa(question.format(author), bio)

    region = "americas"
    if author in europe_bios.keys():
        region = "europe"
    if author in africa_asia_bios.keys():
        region = "africa_asia"
    answer = {"region": region,
              "author": author,
              "answer": response}
    author_answers.append(answer)
    if i % 10 == 0:
        print(f"{i} / {len(bios.keys())} done.")


df = pd.DataFrame(author_answers, columns=["region", "author", "answer"])
df.to_csv(f"./{question}.csv", index=False)
