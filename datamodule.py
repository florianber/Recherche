# Imports
from functions import *
from imports import *
from locations import *
from parameters import *

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device is: ", device)

# Load global dataset
def load_dataset_wikibio(dataset_size) -> Dataset:
    wikibio_dataset = load_dataset("wiki_bio")
    sample_dataset = wikibio_dataset["train"].shuffle(seed=42).select(range(dataset_size))
    
    table_text = []
    for row in sample_dataset:
        if 'table' in row['input_text'] and 'column_header' in row['input_text']['table']:
            header_content_pairs = zip(row['input_text']['table']['column_header'], row['input_text']['table']['content'])
            table_text.append(' '.join([f"{header}: {content}" for header, content in header_content_pairs]) + f" {row['input_text']['context']}")
        else:
            table_text.append(row['input_text']['context'])
    sample_dataset = sample_dataset.remove_columns(['input_text'])
    sample_dataset = sample_dataset.add_column("input_text", table_text)
    return sample_dataset

# Save dataset
def save_dataset(dataset: Dataset, path) -> None:
    dataset.save_to_disk(path)

# Load roberta model
def load_roberta_model():
    return AutoModel.from_pretrained("roberta-base")

# Pre-trained Tokenizer training
def get_trained_tokenizer(dataset: Dataset, dataset_size, vocab_size=50000):
    # Load the roberta pretrained tokenizer
    old_tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    # Prepare the training corpus for the tokenizer
    tokenizer_training_corpus = []
    for i in range(dataset_size):
        tokenizer_training_corpus.append(dataset[i]['input_text'])
        tokenizer_training_corpus.append(dataset[i]['target_text'])

    # Train the tokenizer on the new corpus
    new_tokenizer = old_tokenizer.train_new_from_iterator(tokenizer_training_corpus, vocab_size=vocab_size)

    return new_tokenizer

# Embedding from a pre-trained model
def get_target_embeddings(model, tokens, attention_mask):
    with torch.no_grad():
        # Formatting tokens and attention_mask
        if isinstance(tokens, list):
            tokens = torch.LongTensor([tokens])
            attention_mask = torch.LongTensor([attention_mask])
        else :
            tokens = torch.LongTensor(tokens)
            attention_mask = torch.LongTensor(attention_mask)

        # Pass tokens through the model
        target_outputs = model(input_ids=tokens, attention_mask=attention_mask)

        # Extract embeddings from the last layer
        target_embeddings = target_outputs.last_hidden_state

        # Take the mean of embeddings along the sequence dimension
        target_mean_embeddings = target_embeddings.mean(dim=1).squeeze()

    return {
        "target_embeddings": target_mean_embeddings
    }

# Preprocessing of a sample
def preprocessing(sample, model, tokenizer, index, max_length=512):
    # Tokenize the target_text using the tokenizer
    target_tokens = tokenizer(sample['target_text'], max_length=max_length, padding='max_length', truncation=True, return_tensors='pt', return_attention_mask=True)

    # Tokenize the input_text using the tokenizer
    input_tokens = tokenizer(sample['input_text'], max_length=max_length, padding='max_length', truncation=True, return_tensors='pt', return_attention_mask=True)

    # Embed target_tokens
    target_embeddings = get_target_embeddings(model, target_tokens["input_ids"], target_tokens["attention_mask"])

    return {
        "target_tokens": target_tokens["input_ids"][0],
        "target_attention_mask": target_tokens["attention_mask"][0],
        "target_embeddings": target_embeddings['target_embeddings'],
        "input_tokens": input_tokens["input_ids"][0],
        "input_attention_mask": input_tokens["attention_mask"][0],
        "label": index
    }

# Preprocessing of a dataset
def preprocessing_dataset(dataset: Dataset, model, tokenizer):
    return dataset.map(lambda sample, index: preprocessing(sample, model, tokenizer, index), with_indices=True, batched=False)

# Generation of masked dataset (with/without data augmentation depending of nb_candidates value) (with/without unmasked candidates)
def create_masked_dataset(dataset, model, tokenizer, nb_candidates, with_unmasked_candidates=False):
    masked_dataset_dict = {
        'target_tokens': [],
        'target_attention_mask': [],
        'label': []
    }
    for sample in dataset:
        target_tokens = sample['target_tokens']
        special_tokens = [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id, tokenizer.unk_token_id, tokenizer.mask_token_id]
        masked_indices = [i for i, token_id in enumerate(target_tokens) if token_id not in special_tokens]
        
        for _ in range(nb_candidates):  
            l = random.randint(0, len(masked_indices) - 1)
            masked_indices_to_replace = random.sample(masked_indices, l)
            masked_target_tokens = target_tokens.copy()

            for idx in masked_indices_to_replace:
                masked_target_tokens[idx] = tokenizer.mask_token_id

            masked_dataset_dict['target_tokens'].append(masked_target_tokens)
            masked_dataset_dict['target_attention_mask'].append(sample['target_attention_mask'])
            masked_dataset_dict['label'].append(sample['label'])
        
        if with_unmasked_candidates:
            masked_dataset_dict['target_tokens'].append(target_tokens)
            masked_dataset_dict['target_attention_mask'].append(sample['target_attention_mask'])
            masked_dataset_dict['label'].append(sample['label'])
        

    masked_dataset = Dataset.from_dict(masked_dataset_dict)
    masked_dataset = masked_dataset.map(lambda sample: get_target_embeddings(model, sample['target_tokens'], sample['target_attention_mask']), batched=False)
    return masked_dataset

def main():
    # Getting the Dataset
    dataset = load_dataset_wikibio(dataset_size=DATASET_SIZE)

    # Training a new tokenizer
    trained_tokenizer = get_trained_tokenizer(dataset=dataset, dataset_size=DATASET_SIZE)
    trained_tokenizer.save_pretrained(TOKENIZER_LOCATION)

    # Getting the embedding model 
    roberta_model = load_roberta_model().to(device)

    # Preprocess of the dataset using the new tokenizer
    dataset =  preprocessing_dataset(dataset=dataset, model=roberta_model, tokenizer=trained_tokenizer)
    print_colored(dataset, "red")
    dataset.save_to_disk(UNMASKED_DATASET_LOCATION)

    # Create a masked dataset without unmasked data
    masked_dataset = create_masked_dataset(dataset=dataset, model=roberta_model, tokenizer=trained_tokenizer, nb_candidates=1)
    print_colored(masked_dataset, "blue")
    masked_dataset.save_to_disk(MASKED_WITHOUT_UNMASKED_DATASET_LOCATION)

    # Create a masked dataset with unmasked data
    masked_dataset = create_masked_dataset(dataset=dataset, model=roberta_model, tokenizer=trained_tokenizer, nb_candidates=1, with_unmasked_candidates=True)
    print_colored(masked_dataset, "blue")
    masked_dataset.save_to_disk(MASKED_WITH_UNMASKED_DATASET_LOCATION)

    # Create a masked dataset with data augmentation without unmasked data
    masked_augmented_dataset =  create_masked_dataset(dataset=dataset, model=roberta_model, tokenizer=trained_tokenizer, nb_candidates=AUGMENTATION_RATIO)
    print_colored(masked_augmented_dataset, "green")
    masked_augmented_dataset.save_to_disk(AUGMENTED_MASKED_WITHOUT_UNMASKED_DATASET_LOCATION)

    # Create a masked dataset with data augmentation with unmasked data
    masked_augmented_dataset =  create_masked_dataset(dataset=dataset, model=roberta_model, tokenizer=trained_tokenizer, nb_candidates=AUGMENTATION_RATIO, with_unmasked_candidates=True)
    print_colored(masked_augmented_dataset, "green")
    masked_augmented_dataset.save_to_disk(AUGMENTED_MASKED_WITH_UNMASKED_DATASET_LOCATION)

if __name__ == '__main__':
    main()