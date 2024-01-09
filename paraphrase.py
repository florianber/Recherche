import torch

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch



device = "cpu"
tokenizer = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")

model = AutoModelForSeq2SeqLM.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base").to(device)

# def paraphrase(
#     question,
#     num_beams=5,
#     num_beam_groups=5,
#     num_return_sequences=1,
#     repetition_penalty=10.0,
#     diversity_penalty=3.0,
#     no_repeat_ngram_size=2,
#     max_length=128
# ):

def paraphrase(
    question,
    num_beams=5,
    num_beam_groups=5,
    num_return_sequences=1,
    repetition_penalty=5.0,
    diversity_penalty=5.0,
    no_repeat_ngram_size=5,
    max_length=200
):
    input_ids = tokenizer(
        f'paraphrase: {question}',
        return_tensors="pt", padding="longest",
        max_length=max_length,
        truncation=True,
    ).input_ids.to(device)

    outputs = model.generate(
        input_ids, repetition_penalty=repetition_penalty,
        num_return_sequences=num_return_sequences, no_repeat_ngram_size=no_repeat_ngram_size,
        num_beams=num_beams, num_beam_groups=num_beam_groups,
        max_length=max_length, diversity_penalty=diversity_penalty
    )

    res = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return res


text = "leonard shenoff randle -lrb- born february 12 , 1949 -rrb- is a former major league baseball player . he was the first-round pick of the washington senators in the secondary phase of the june 1970 major league baseball draft , tenth overall ."
paraphrased=paraphrase(text)
print(paraphrased[0].lower())
"in the june 1970 major league baseball draft, leonard shenoff randle, who was born on february 12, 1949, and lived in rural vermont, was selected as the first-round pick by the washington senators and finished tenth overall."