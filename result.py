# Imports
from functions import *
from imports import *
from locations import *
from parameters import *

# Loading the datasets
unmasked_dataset = load_from_disk(UNMASKED_DATASET_LOCATION)
masked_with_unmasked_dataset = load_from_disk(MASKED_WITH_UNMASKED_DATASET_LOCATION)
masked_without_unmasked_dataset = load_from_disk(MASKED_WITHOUT_UNMASKED_DATASET_LOCATION)
augmented_masked_with_unmasked_dataset = load_from_disk(AUGMENTED_MASKED_WITH_UNMASKED_DATASET_LOCATION)
augmented_masked_without_unmasked_dataset = load_from_disk(AUGMENTED_MASKED_WITHOUT_UNMASKED_DATASET_LOCATION)
ner_dataset = load_from_disk(NER_DATASET_LOCATION)
lexical_dataset = load_from_disk(LEXICAL_DATASET_LOCATION)
idf_dataset = load_from_disk(IDF_DATASET_LOCATION)
idf_table_aware_dataset = load_from_disk(IDF_TABLE_AWARE_DATASET_LOCATION)

# region Model's objects:
class NN_1024_768_300_EMB(nn.Module):
    def __init__(self, dataset: Dataset, emb_field):
        super(NN_1024_768_300_EMB, self).__init__()
        self.inputs = self.get_inputs(dataset, emb_field)
        self.input_size = self.get_input_size(dataset, emb_field)
        self.labels = self.get_labels(dataset)
        self.output_size = len(self.labels)
        self.losses = []
        self.accuracies = []

        # Define layers for classification
        self.fc1 = nn.Linear(self.input_size, 1024)
        self.fc2 = nn.Linear(1024, 768)
        self.fc3 = nn.Linear(768, 300)
        self.fc4 = nn.Linear(300, self.output_size)

    
    def forward(self, x):
      x = self.fc1(x)
      x = self.fc2(x)
      x = torch.nn.functional.relu(self.fc3(x))
      x = self.fc4(x)
      return x
    
    def get_inputs(self, dataset: Dataset, emb_field):
        return [sample[emb_field] for sample in dataset]

    def get_labels(self, dataset: Dataset):
        labels = set(sample['label'] for sample in dataset)
        return list(labels)
    

    def get_input_size(self, dataset: Dataset, emb_field):
        if len(dataset) == 0:
            raise ValueError("The dataset is empty")

        first_instance_size = len(dataset[0][emb_field])
        for instance in dataset:
            current_size = len(instance[emb_field])
            if current_size != first_instance_size:
                raise ValueError("Sizes of target_embeddings are not consistent across the entire dataset")

        return first_instance_size
    
class NN_1024_768_300_IDS(nn.Module):
    def __init__(self, dataset: Dataset, token_field):
        super(NN_1024_768_300_IDS, self).__init__()
        self.inputs = self.get_inputs(dataset, token_field)
        self.input_size = self.get_input_size(dataset, token_field)
        self.labels = self.get_labels(dataset)
        self.output_size = len(self.labels)
        self.losses = []
        self.accuracies = []

        # Define layers for classification
        self.fc1 = nn.Linear(self.input_size, 1024)
        self.fc2 = nn.Linear(1024, 768)
        self.fc3 = nn.Linear(768, 300)
        self.fc4 = nn.Linear(300, self.output_size)

    def forward(self, x):
      x = x.float()
      x = self.fc1(x)
      x = self.fc2(x)
      x = self.fc3(x)
      x = self.fc4(x)
      return x
    
    def get_inputs(self, dataset: Dataset, token_field):
        return [sample[token_field] for sample in dataset]

    def get_labels(self, dataset: Dataset):
        labels = set(sample['label'] for sample in dataset)
        return list(labels)

    def get_input_size(self, dataset: Dataset, token_field):
        if len(dataset) == 0:
            raise ValueError("The dataset is empty")

        first_instance_size = len(dataset[0][token_field])
        for instance in dataset:
            current_size = len(instance[token_field])
            if current_size != first_instance_size:
                raise ValueError("Sizes of target_embeddings are not consistent across the entire dataset")

        return first_instance_size

class NN_1024_1024_EMB(nn.Module):
    def __init__(self, dataset: Dataset, emb_field):
        super(NN_1024_1024_EMB, self).__init__()
        self.inputs = self.get_inputs(dataset, emb_field)
        self.input_size = self.get_input_size(dataset, emb_field)
        self.labels = self.get_labels(dataset)
        self.output_size = len(self.labels)
        self.losses = []
        self.accuracies = []

        # Define layers for classification
        self.fc1 = nn.Linear(self.input_size, 1024)
        self.fc2 = nn.Linear(1024, self.output_size)


    def forward(self, x):
      x = self.fc1(x)
      x = self.fc2(x)
      return x
    
    def get_inputs(self, dataset: Dataset, emb_field):
        return [sample[emb_field] for sample in dataset]

    def get_labels(self, dataset: Dataset):
        labels = set(sample['label'] for sample in dataset)
        return list(labels)
    

    def get_input_size(self, dataset: Dataset, emb_field):
        if len(dataset) == 0:
            raise ValueError("The dataset is empty")

        first_instance_size = len(dataset[0][emb_field])
        for instance in dataset:
            current_size = len(instance[emb_field])
            if current_size != first_instance_size:
                raise ValueError("Sizes of target_embeddings are not consistent across the entire dataset")

        return first_instance_size
# endregion

def mask_percentages(datasets):
    percentages_dataset = {}
    for dataset, name, token_field, _ in datasets:
        percentages = []
        for sample in dataset:
            masked_token_count = 0
            token_count = 0
            for token in sample[token_field]:
                token_count += 1
                if token == 1:
                    break
                if token == 4:
                    masked_token_count += 1
            percentages.append(masked_token_count / token_count)
        percentages_dataset[name] = np.mean(percentages)

    return percentages_dataset

def compute_metrics(true_labels, predicted_labels):
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='micro')
    recall = recall_score(true_labels, predicted_labels, average='micro')
    f1 = f1_score(true_labels, predicted_labels, average='micro')
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    return accuracy, precision, recall, f1, conf_matrix

def get_results(model, dataset, token_field, emb_field, emb_model=True):

    if emb_model :
        input_field = emb_field
        Dtype = torch.float32

    else :
        input_field = token_field
        Dtype = torch.int64

    model.eval()
    true_labels, predicted_labels = [], []
    for sample in dataset:
        input = sample[input_field]
        label = sample['label']
        
        with torch.no_grad():
            output = model(torch.tensor(input, dtype=Dtype))

        prediction = np.argmax(output.numpy())
        true_labels.append(label)
        predicted_labels.append(prediction)

    # Compute metrics
    accuracy, precision, recall, f1, conf_matrix = compute_metrics(true_labels, predicted_labels)

    return accuracy, precision, recall, f1, conf_matrix


def put_in_file(filename,name,res,model_location):

    with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            
            # Writing new rows
            writer.writerow([
            f"Dataset name: {name}",
            f"Model: {model_location}",
            f"Accuracy: {res[0]}",
            f"Precision: {res[1]}",
            f"Recall: {res[2]}",
            f"F1 Score: {res[3]}",
            f"Confusion Matrix:\n{res[4]}\n"
        ])


def main():

    datasets = [
        (unmasked_dataset, "unmasked_dataset", 'target_tokens', 'target_embeddings'),
        (masked_with_unmasked_dataset, "masked_with_unmasked_dataset", 'target_tokens', 'target_embeddings'),
        (masked_without_unmasked_dataset, "masked_without_unmasked_dataset", 'target_tokens', 'target_embeddings'),
        (augmented_masked_with_unmasked_dataset, "augmented_masked_with_unmasked_dataset", 'target_tokens', 'target_embeddings'),
        (augmented_masked_without_unmasked_dataset, "augmented_masked_without_unmasked_dataset", 'target_tokens', 'target_embeddings'),
        (ner_dataset, "ner_dataset", 'ner_tokens', 'ner_embeddings'),
        (lexical_dataset, "lexical_dataset", 'lexical_tokens', 'lexical_embeddings'),
        (idf_dataset, "idf_dataset", 'idf_tokens', 'idf_embeddings'),
        (idf_table_aware_dataset, "idf_table_aware_dataset", 'idf_table_aware_tokens', 'idf_table_aware_embeddings')
    ]

    datasets_mask_percentages = mask_percentages(datasets)
    print_colored(datasets_mask_percentages, "red")

    filename = 'result1.csv'
    with open(filename, mode='w', newline='') as file:
        pass

    for dataset, name , token_field, emb_field in datasets: 

        # nn_1024_768_300_emb
        print("Calculating results for nn_1024_768_300_emb model ...")
        nn_1024_768_300_emb_unmasked = NN_1024_768_300_EMB(dataset=dataset, emb_field=emb_field)
        nn_1024_768_300_emb_unmasked.load_state_dict(torch.load(NN_1024_768_300_EMB_UNMASKED_REID_MODEL))
        nn_1024_768_300_emb_unmasked_res = get_results(nn_1024_768_300_emb_unmasked, dataset, token_field, emb_field)

        # Writing data to the CSV file
        put_in_file(filename,name,nn_1024_768_300_emb_unmasked_res,NN_1024_768_300_EMB_UNMASKED_REID_MODEL)

        nn_1024_768_300_emb_masked_with_unmasked = NN_1024_768_300_EMB(dataset=dataset, emb_field=emb_field)
        nn_1024_768_300_emb_masked_with_unmasked.load_state_dict(torch.load(NN_1024_768_300_EMB_MASKED_WITH_UNMASKED_REID_MODEL))
        nn_1024_768_300_emb_masked_with_unmasked_res = get_results(nn_1024_768_300_emb_masked_with_unmasked, dataset, token_field, emb_field)

        put_in_file(filename,name,nn_1024_768_300_emb_masked_with_unmasked_res,NN_1024_768_300_EMB_MASKED_WITH_UNMASKED_REID_MODEL)

        nn_1024_768_300_emb_masked_without_unmasked = NN_1024_768_300_EMB(dataset=dataset, emb_field=emb_field)
        nn_1024_768_300_emb_masked_without_unmasked.load_state_dict(torch.load(NN_1024_768_300_EMB_MASKED_WITHOUT_UNMASKED_REID_MODEL))
        nn_1024_768_300_emb_masked_without_unmasked_res = get_results(nn_1024_768_300_emb_masked_without_unmasked, dataset, token_field, emb_field)

        put_in_file(filename,name,nn_1024_768_300_emb_masked_without_unmasked_res,NN_1024_768_300_EMB_MASKED_WITHOUT_UNMASKED_REID_MODEL)
        
        nn_1024_768_300_emb_augmented_masked_with_unmasked = NN_1024_768_300_EMB(dataset=dataset, emb_field=emb_field)
        nn_1024_768_300_emb_augmented_masked_with_unmasked.load_state_dict(torch.load(NN_1024_768_300_EMB_AUGMENTED_MASKED_WITH_UNMASKED_REID_MODEL))
        nn_1024_768_300_emb_augmented_masked_with_unmasked_res = get_results(nn_1024_768_300_emb_augmented_masked_with_unmasked, dataset, token_field, emb_field)
        
        put_in_file(filename,name,nn_1024_768_300_emb_augmented_masked_with_unmasked_res,NN_1024_768_300_EMB_AUGMENTED_MASKED_WITH_UNMASKED_REID_MODEL)
            
        nn_1024_768_300_emb_augmented_masked_without_unmasked = NN_1024_768_300_EMB(dataset=dataset, emb_field=emb_field)
        nn_1024_768_300_emb_augmented_masked_without_unmasked.load_state_dict(torch.load(NN_1024_768_300_EMB_AUGMENTED_MASKED_WITHOUT_UNMASKED_REID_MODEL))
        nn_1024_768_300_emb_augmented_masked_without_unmasked_res = get_results(nn_1024_768_300_emb_augmented_masked_without_unmasked, dataset, token_field, emb_field)

        put_in_file(filename,name,nn_1024_768_300_emb_augmented_masked_without_unmasked_res,NN_1024_768_300_EMB_AUGMENTED_MASKED_WITHOUT_UNMASKED_REID_MODEL)

        # nn_1024_768_300_ids
        print("Calculating results for nn_1024_768_300_ids model ...")
        nn_1024_768_300_ids_unmasked = NN_1024_768_300_IDS(dataset=dataset, token_field=token_field)
        nn_1024_768_300_ids_unmasked.load_state_dict(torch.load(NN_1024_768_300_IDS_UNMASKED_REID_MODEL))
        nn_1024_768_300_ids_unmasked_res = get_results(nn_1024_768_300_ids_unmasked, dataset, token_field, emb_field, emb_model=False)

        put_in_file(filename,name,nn_1024_768_300_ids_unmasked_res,NN_1024_768_300_IDS_UNMASKED_REID_MODEL)

        nn_1024_768_300_ids_masked_with_unmasked = NN_1024_768_300_IDS(dataset=dataset, token_field=token_field)
        nn_1024_768_300_ids_masked_with_unmasked.load_state_dict(torch.load(NN_1024_768_300_IDS_MASKED_WITH_UNMASKED_REID_MODEL))
        nn_1024_768_300_ids_masked_with_unmasked_res = get_results(nn_1024_768_300_ids_masked_with_unmasked, dataset, token_field, emb_field, emb_model=False)

        put_in_file(filename,name,nn_1024_768_300_ids_masked_with_unmasked_res,NN_1024_768_300_IDS_MASKED_WITH_UNMASKED_REID_MODEL)
            
        nn_1024_768_300_ids_masked_without_unmasked = NN_1024_768_300_IDS(dataset=dataset, token_field=token_field)
        nn_1024_768_300_ids_masked_without_unmasked.load_state_dict(torch.load(NN_1024_768_300_IDS_MASKED_WITHOUT_UNMASKED_REID_MODEL))
        nn_1024_768_300_ids_masked_without_unmasked_res = get_results(nn_1024_768_300_ids_masked_without_unmasked, dataset, token_field, emb_field, emb_model=False)
        
        put_in_file(filename,name,nn_1024_768_300_ids_masked_without_unmasked_res,NN_1024_768_300_IDS_MASKED_WITHOUT_UNMASKED_REID_MODEL)
            
        nn_1024_768_300_ids_augmented_masked_with_unmasked = NN_1024_768_300_IDS(dataset=dataset, token_field=token_field)
        nn_1024_768_300_ids_augmented_masked_with_unmasked.load_state_dict(torch.load(NN_1024_768_300_IDS_AUGMENTED_MASKED_WITH_UNMASKED_REID_MODEL))
        nn_1024_768_300_ids_augmented_masked_with_unmasked_res = get_results(nn_1024_768_300_ids_augmented_masked_with_unmasked, dataset, token_field, emb_field, emb_model=False)
        
        put_in_file(filename,name,nn_1024_768_300_ids_augmented_masked_with_unmasked_res,NN_1024_768_300_IDS_AUGMENTED_MASKED_WITH_UNMASKED_REID_MODEL)
            
        nn_1024_768_300_ids_augmented_masked_without_unmasked = NN_1024_768_300_IDS(dataset=dataset, token_field=token_field)
        nn_1024_768_300_ids_augmented_masked_without_unmasked.load_state_dict(torch.load(NN_1024_768_300_IDS_AUGMENTED_MASKED_WITHOUT_UNMASKED_REID_MODEL))
        nn_1024_768_300_ids_augmented_masked_without_unmasked_res = get_results(nn_1024_768_300_ids_augmented_masked_without_unmasked, dataset, token_field, emb_field, emb_model=False)

        put_in_file(filename,name,nn_1024_768_300_ids_augmented_masked_without_unmasked_res,NN_1024_768_300_IDS_AUGMENTED_MASKED_WITHOUT_UNMASKED_REID_MODEL)
            
        """
        # nn_1024_1024_emb
        print("Calculating results for nn_1024_1024_emb model ...")
        state_dict = torch.load(NN_1024_1024_EMB_MASKED_WITHOUT_UNMASKED, map_location=torch.device('cpu'))

        # Imprimer les noms de toutes les couches
        for key in state_dict.keys():
            print(key)
            
        nn_1024_1024_emb_unmasked = NN_1024_1024_EMB(dataset=dataset, emb_field=emb_field)
        nn_1024_1024_emb_unmasked.load_state_dict(torch.load(NN_1024_1024_EMB_UNMASKED))
        nn_1024_1024_emb_unmasked_res = get_results(nn_1024_1024_emb_unmasked, dataset, token_field, emb_field)
        
        nn_1024_1024_emb_masked_without_unmasked = NN_1024_1024_EMB(dataset=dataset, emb_field=emb_field)
        nn_1024_1024_emb_masked_without_unmasked.load_state_dict(torch.load(NN_1024_1024_EMB_MASKED_WITHOUT_UNMASKED))
        nn_1024_1024_emb_masked_without_unmasked_res = get_results(nn_1024_1024_emb_masked_without_unmasked, dataset, token_field, emb_field)
        
        nn_1024_1024_emb_masked_with_unmasked = NN_1024_1024_EMB(dataset=dataset, emb_field=emb_field)
        nn_1024_1024_emb_masked_with_unmasked.load_state_dict(torch.load(NN_1024_1024_EMB_AUGMENTED_MASKED_WITHOUT_UNMASKED))
        nn_1024_1024_emb_masked_with_unmasked_res = get_results(nn_1024_1024_emb_masked_with_unmasked, dataset, token_field, emb_field)
        """
if __name__ == "__main__":
    main()