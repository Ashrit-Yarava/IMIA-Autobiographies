import json
from pipeline import QuestionAnswering


americas_bios = json.loads(open("../data/jsons/americas.json").read())
europe_bios = json.loads(open("../data/jsons/europe.json").read())
africa_asia_bios = json.loads(open("../data/jsons/africaasia.json").read())

bios = {**americas_bios, **europe_bios, **africa_asia_bios}


AUTHOR = "Jos Aarts"
bio = "I am Jos Aarts. " + bios[AUTHOR]
QUESTION = "What did Jos Aarts major in?"


model, tokenizer = QuestionAnswering.load_model()
qa = QuestionAnswering(model, tokenizer)

response = qa(QUESTION, bio)
print(response)
