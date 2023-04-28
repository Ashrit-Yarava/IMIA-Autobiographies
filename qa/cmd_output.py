import argparse
import json
from pipeline import QuestionAnswering


def load_biographies():
    """
    Returns a dictionaries of all the biogrpaheis of the authors
    as a dictionary.
    """
    DIR_NAME = "./data/jsons/"

    americas_bios = json.loads(open(f"{DIR_NAME}americas.json").read())
    europe_bios = json.loads(open(f"{DIR_NAME}europe.json").read())
    africa_asia_bios = json.loads(open(f"{DIR_NAME}africaasia.json").read())

    bios = {**americas_bios, **europe_bios, **africa_asia_bios}

    return bios


def main(author, question):
    """
    Main function to display the output and process the question using the T5 Model.
    """
    bios = load_biographies()
    bio_to_use = f"I am {author}. " + bios[author]

    qa = QuestionAnswering(*QuestionAnswering.load_model())

    output = qa(question, bio_to_use)

    print(output)
    return output


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Script to run a question against one author.")
    parser.add_argument("author", type=str, required=True, help="The author's autobiography to use.")
    parser.add_argument("question", type=str, required=True, help="The question to ask.")

    args = parser.parse_args()

    main(args.author, args.question)
