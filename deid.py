# Imports
from datamodule import load_roberta_model, save_dataset
from functions import *
from imports import *
from locations import *
from parameters import *

# Loading SpaCy NER pre-trained ("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")

# Corpus for IDF methods
def get_corpus(dataset: Dataset):
    corpus = [sample["target_text"].replace("\n", "") + sample["input_text"].replace("\n", "") for sample in dataset]
    return corpus

# Embedding from a pre-trained model
def get_embeddings(model, tokens, attention_mask):
    with torch.no_grad():
        # Formatting tokens and attention_mask
        if isinstance(tokens, list):
            tokens = torch.LongTensor([tokens])
            attention_mask = torch.LongTensor([attention_mask])
        else:
            tokens = torch.LongTensor(tokens)
            attention_mask = torch.LongTensor(attention_mask)

        # Pass tokens through the model
        outputs = model(input_ids=tokens, attention_mask=attention_mask)

        # Extract embeddings from the last layer
        embeddings = outputs.last_hidden_state

        # Take the mean of embeddings along the sequence dimension
        mean_embeddings = embeddings.mean(dim=1).squeeze()

    return {
        "embeddings": mean_embeddings
    }

# Preprocessing of a sample
def preprocessing(sample, model, tokenizer, name, index, max_length=512):
    # Tokenize
    if str(name) == "ner":
        tokens = tokenizer(sample['ner_text'], max_length=max_length, padding='max_length', truncation=True, return_tensors='pt', return_attention_mask=True)
    elif str(name) == "lexical":
        tokens = tokenizer(sample['lexical_text'], max_length=max_length, padding='max_length', truncation=True, return_tensors='pt', return_attention_mask=True)
    elif str(name) == "idf":
        tokens = tokenizer(sample['idf_text'], max_length=max_length, padding='max_length', truncation=True, return_tensors='pt', return_attention_mask=True)
    elif str(name) == "idf_table_aware":
        tokens = tokenizer(sample['idf_table_aware_text'], padding='max_length', truncation=True, return_tensors='pt', return_attention_mask=True)

    # Embed tokens
    embeddings = get_embeddings(model, tokens["input_ids"], tokens["attention_mask"])

    return {
        str(name)+"_tokens": tokens["input_ids"][0],
        str(name)+"_attention_mask": tokens["attention_mask"][0],
        str(name)+"_embeddings": embeddings['embeddings'],
        "label": index
    }

# Preprocessing of a dataset
def preprocessing_dataset(dataset: Dataset, model, tokenizer, name):
    return dataset.map(lambda sample, index: preprocessing(sample, model=model, tokenizer=tokenizer, name=name, index=index), with_indices=True, batched=False)

# NER
def ner(text):
    doc = nlp(text)
    deidentified_text = []

    for token in doc:
        # Masking the text of named entities with a certain probability
        if token.ent_type_:
            deidentified_text.append("<mask>")
        else:
            deidentified_text.append(token.text)

    ner_text = " ".join(deidentified_text)

    return {"ner_text": ner_text}

# LEXICAL
def lexical(text, table_text):
    doc = nlp(text)
    deidentified_text = []
    for token in doc:
        if str(token) in table_text and str(token) not in string.punctuation and str(token) != "\n":
            deidentified_text.append("<mask>")
        else:
            deidentified_text.append(token.text)

    lexical_text = " ".join(deidentified_text)

    return {"lexical_text": lexical_text}

# IDF
def idf(text, corpus):
    # Tokenization
    tokenized_text = simple_preprocess(text.replace("\n", "").replace("-lrb-", "").replace("-rrb-", ""))
    tokenized_corpus = [simple_preprocess(doc) for doc in corpus]

    # Create a Gensim Dictionary and Corpus and Text
    dct = Dictionary(tokenized_corpus)  # fit dictionary
    corpus_bow = [dct.doc2bow(doc) for doc in tokenized_corpus]
    text_bow = dct.doc2bow(tokenized_text)

    # TF-IDF model
    tfidf_model = TfidfModel(corpus_bow)

    # Applying the TF-IDF model
    tfidf_vector = tfidf_model[text_bow]
    token_tfidf_dict = dict(tfidf_vector)

    # Mask the text based on IDF values
    masked_text = []
    for token in tokenized_text:
        token_index = dct.token2id.get(token, -1)
        tfidf_value = token_tfidf_dict.get(token_index, 0.0)
        if tfidf_value < IDF_THRESHOLD:
            masked_text.append("<mask>")
        else:
            masked_text.append(token)

    return {"idf_text": " ".join(masked_text)}

# IDF-Table aware
def idf_table_aware(text, profile, corpus):
    # Tokenization
    tokenized_text = simple_preprocess(text.replace("\n", "").replace("-lrb-", "").replace("-rrb-", ""))
    tokenized_profile = simple_preprocess(profile)
    tokenized_corpus = [simple_preprocess(doc) for doc in corpus]

    # Create a Gensim Dictionary and Corpus
    dct = Dictionary(tokenized_corpus)
    corpus_bow = [dct.doc2bow(doc) for doc in tokenized_corpus]

    # TF-IDF model
    tfidf_model = TfidfModel(corpus_bow)

    # Applying the TF-IDF model
    tfidf_vector = tfidf_model[dct.doc2bow(tokenized_text)]
    token_tfidf_dict = dict(tfidf_vector)

    # Mask the text based on IDF values and overlapping words
    masked_text = []
    for token in tokenized_text:
        token_index = dct.token2id.get(token, -1)
        tfidf_value = token_tfidf_dict.get(token_index, 0.0)
        if token in tokenized_profile or tfidf_value < IDF_TABLE_AWARE_THRESHOLD:
            masked_text.append("<mask>")
        else:
            masked_text.append(token)

    return {"idf_table_aware_text": " ".join(masked_text)}

# Calling each method of deidentification
def deid_dataset(dataset, model, tokenizer, corpus):
    # NER
    print("Generating NER dataset...")
    ner_dataset = dataset.map(lambda sample: ner(text=sample["target_text"]))
    ner_dataset = preprocessing_dataset(dataset=ner_dataset, model=model, tokenizer=tokenizer, name="ner")
    save_dataset(NER_DATASET_LOCATION)
    print_colored(ner_dataset, "red")

    # Lexical
    print("Generating LEXICAL dataset...")
    lexical_dataset = dataset.map(lambda sample: lexical(sample["target_text"], sample['input_text']))
    lexical_dataset = preprocessing_dataset(dataset=lexical_dataset,  model=model, tokenizer=tokenizer, name="lexical")
    save_dataset(LEXICAL_DATASET_LOCATION)
    print_colored(lexical_dataset, "blue")

    # IDF
    print("Generating IDF dataset...")
    idf_dataset = dataset.map(lambda sample: idf(sample["target_text"], corpus))
    idf_dataset = preprocessing_dataset(dataset=idf_dataset,  model=model, tokenizer=tokenizer, name="idf")
    save_dataset(IDF_DATASET_LOCATION)
    print_colored(idf_dataset, "green")

    # IDF-table aware
    print("Generating IDF-table-aware dataset...")
    idf_table_aware_dataset = dataset.map(lambda sample: idf_table_aware(sample["target_text"], sample['input_text'], corpus))
    idf_table_aware_dataset = preprocessing_dataset(dataset=idf_table_aware_dataset,  model=model, tokenizer=tokenizer, name="idf_table_aware")
    save_dataset(IDF_TABLE_AWARE_DATASET_LOCATION)
    print_colored(idf_table_aware_dataset, "red")

def main():
    # Load the RoBERTa model and the tokenizer
    roberta_model = load_roberta_model()
    pretrained_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=TOKENIZER_LOCATION)

    # Load the unmasked dataset
    unmasked_dataset = load_from_disk(dataset_path=UNMASKED_DATASET_LOCATION)
    
    # Copy the dataset to a new instance
    copied_unmasked_dataset = Dataset.from_dict(unmasked_dataset.to_dict())

    # Get the corpus for IDF methods
    corpus = get_corpus(dataset=copied_unmasked_dataset)

    # Creation of one dataset for each deid method
    deid_dataset(dataset=unmasked_dataset, model=roberta_model, tokenizer=pretrained_tokenizer, corpus=corpus)

if __name__ == '__main__':
    main()
