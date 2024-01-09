# Imports
from datamodule import load_roberta_model
from functions import *
from imports import *
from locations import *
from parameters import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device is: ", device)

# Getting embeddings
def get_embeddings(target_model, input_model, target_tokens, target_attention_mask, input_tokens, input_attention_mask):
    with torch.no_grad():
        # Pass tokens through the model
        target_outputs = target_model(input_ids=torch.LongTensor(target_tokens), attention_mask=torch.LongTensor(target_attention_mask))
        input_outputs = input_model(input_ids=torch.LongTensor(input_tokens), attention_mask=torch.LongTensor(input_attention_mask))

        # Extract embeddings from the last layer
        target_embeddings = target_outputs.last_hidden_state
        input_embeddings = input_outputs.last_hidden_state

        # Take the mean of embeddings along the sequence dimension
        target_mean_embeddings = target_embeddings.mean(dim=1).squeeze()
        input_mean_embeddings = input_embeddings.mean(dim=1).squeeze()

    return {
        "target_embeddings": target_mean_embeddings,
        "input_embeddings": input_mean_embeddings
    }

# Updating Embeddings
def update_embeddings(target_model, input_model, dataset: Dataset):
    return dataset.map(lambda sample: get_embeddings(target_model, input_model, sample['target_tokens'], sample['target_attention_mask'], sample['input_tokens'], sample['input_attention_mask']), batched=False)

# Saving a pretrained model
def save_model(model, location):
    empty_folder(location)
    return model.save_pretrained(location)

# Getting our optimizer
def get_optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=0.001)

# region : Loss 
""" 
# Idea : Calculates the loss -> The more distance pos/neg, the best  
def multi_loss(anchor, similarities, positive_index, margin=1.0):
    loss = torch.Tensor([])    
    pos_distance = F.pairwise_distance(anchor.unsqueeze(0), similarities[positive_index].unsqueeze(0))
    for i, similarity in enumerate(similarities):
        if i != positive_index:
            neg_distance = F.pairwise_distance(anchor.unsqueeze(0), similarity.unsqueeze(0))
            loss = torch.cat((loss, torch.relu(neg_distance - pos_distance + margin)))

    return loss.mean()
"""

# Calculates the loss -> The less of (neg_similarities - pos_similarity), the best
def multi_loss(similarities, positive_index):
    neg_similarities = torch.cat((similarities[:positive_index], similarities[positive_index + 1:]))
    loss = torch.sum(neg_similarities) - similarities[positive_index]
    return loss
# endregion 

# region : Similarity 
def calculate_similarities(target_embedding, input_embeddings):
    similarities = torch.nn.functional.cosine_similarity(target_embedding.unsqueeze(0), input_embeddings, dim=1)
    return similarities

"""
def calculate_similarity(target_embedding, input_embedding):
    # Calcule la similarité cosinus
    similarity = torch.nn.functional.cosine_similarity(target_embedding, input_embedding, dim=0)

    return similarity
"""
# endregion 

# Train the model, minimizing the similarity loss
def train(optimizer, dataset: Dataset):
    losses = []

    input_embeddings_list = []
    for entry in dataset:
        input_embedding = torch.FloatTensor(entry['input_embeddings']).to(device).requires_grad_(True)
        input_embeddings_list.append(input_embedding)

    input_embeddings = torch.stack(input_embeddings_list)

    for positive_index, sample in enumerate(dataset):
        target_embedding = torch.FloatTensor(sample['target_embeddings']).to(device).requires_grad_(True)
        input_embedding = torch.FloatTensor(sample['input_embeddings']).to(device).requires_grad_(True)

        # Calculate similarities
        similarities = calculate_similarities(target_embedding, input_embeddings)
        print_colored(similarities, "red")

        # Similarity-based loss
        loss = multi_loss(similarities, positive_index)
        print_colored(loss, "blue")
        losses.append(loss)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    total_loss = torch.sum(torch.stack(losses))
    print_colored(total_loss, "green")

# Calls the train() function, in order to train target and input models each at a time
def coordinate_ascent_optimization(target_model, input_model, target_optimizer, input_optimizer, dataset: Dataset):
    for epoch in range(NB_EPOCHS):
        print_colored(epoch, "red")
        dataset = update_embeddings(target_model, input_model, dataset)
        if epoch % 10 : # Input model
            train(input_optimizer, dataset)
            save_model(model=input_model, location=PRETRAINED_INPUT_MODEL_LOCATION)
            save_model(model=target_model, location=PRETRAINED_TARGET_MODEL_LOCATION)
        else : # Train the target model
            train(target_optimizer, dataset)


def main():
    # Getting the dataset
    unmasked_dataset = load_from_disk(UNMASKED_DATASET_LOCATION)

    # Loading models
    target_model = load_roberta_model().to(device) # to use gpu
    input_model = load_roberta_model().to(device) # to use gpu

    # Putting models in training mode in order to fine-tune
    target_model.train()
    input_model.train()

    # Getting optimizers for each model
    target_optimizer = get_optimizer(target_model)
    input_optimizer = get_optimizer(input_model)

    # Coordinate ascent optimization : training of target_model (computes the embedding on target_text) and input_model(computes the embedding on input_text)
    coordinate_ascent_optimization(target_model, input_model, target_optimizer, input_optimizer, unmasked_dataset)

if __name__ == '__main__':
    main()
