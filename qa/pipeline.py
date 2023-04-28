from transformers import AutoModelWithLMHead, AutoTokenizer


class QuestionAnswering:

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, question: str, context: str):
        prompt = f"question: {question} context: {context}"
        encoded_input = self.tokenizer([prompt], return_tensors='pt', max_length=512, truncation=True)
        output = self.model.generate(input_ids=encoded_input.input_ids, attention_mask=encoded_input.attention_mask)
        output = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return output

    @staticmethod
    def load_model():
        model_name = "MaRiOrOsSi/t5-base-finetuned-question-answering"
        model = AutoModelWithLMHead.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model, tokenizer


if __name__ == "__main__":
    # Download and load the model.
    model_name = "MaRiOrOsSi/t5-base-finetuned-question-answering"
    model = AutoModelWithLMHead.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    qa = QuestionAnswering(model, tokenizer)

    context = "Batman is a superhero appearing in American comic books published by DC Comics. The character was created by artist Bob Kane and writer Bill Finger, and debuted in the 27th issue of the comic book Detective Comics on March 30, 1939. In the DC Universe continuity, Batman is the alias of Bruce Wayne, a wealthy American playboy, philanthropist, and industrialist who resides in Gotham City. Batman's origin story features him swearing vengeance against criminals after witnessing the murder of his parents Thomas and Martha as a child, a vendetta tempered with the ideal of justice. He trains himself physically and intellectually, crafts a bat-inspired persona, and monitors the Gotham streets at night. Kane, Finger, and other creators accompanied Batman with supporting characters, including his sidekicks Robin and Batgirl; allies Alfred Pennyworth, James Gordon, and Catwoman; and foes such as the Penguin, the Riddler, Two-Face, and his archenemy, the Joker."
    question = "Who is Batman?"

    output = qa(question, context)
    open("temp.txt", "w").write(f"{question}\n{output}")
