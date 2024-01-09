# Example of script that we ran to train our models
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

class NN_1024_768_300_EMB(nn.Module):
    def __init__(self, dataset: Dataset):
        super(NN_1024_768_300_EMB, self).__init__()
        self.inputs = self.get_inputs(dataset)
        self.input_size = self.get_input_size(dataset)
        self.labels = self.get_labels(dataset)
        self.output_size = len(self.labels)
        self.losses = []
        self.accuracies = []

        # Define layers for classification
        self.fc1 = nn.Linear(self.input_size, NN_1024_768_300_N1)
        self.fc2 = nn.Linear(NN_1024_768_300_N1, NN_1024_768_300_N2)
        self.fc3 = nn.Linear(NN_1024_768_300_N2, NN_1024_768_300_N3)
        self.fc4 = nn.Linear(NN_1024_768_300_N3, self.output_size)

    # Get the inputs
    def get_inputs(self, dataset: Dataset):
        return [sample['target_embeddings'] for sample in dataset]

    # Get the labels
    def get_labels(self, dataset: Dataset):
        labels = set(sample['label'] for sample in dataset)
        return list(labels)

    # Get the input size
    def get_input_size(self, dataset: Dataset):
        if len(dataset) == 0:
            raise ValueError("The dataset is empty")

        first_instance_size = len(dataset[0]['target_embeddings'])
        for instance in dataset:
            current_size = len(instance['target_embeddings'])
            if current_size != first_instance_size:
                raise ValueError("Sizes of target_embeddings are not consistent across the entire dataset")

        return first_instance_size

    # Forward function (we can change here layers)
    def forward(self, x):
      x = self.fc1(x)
      x = self.fc2(x)
      x = torch.nn.functional.relu(self.fc3(x))
      x = self.fc4(x)
      return x

    # Loss function designed for multi-class classification problems: combines a log-softmax function and the negative log-likelihood loss
    def compute_loss(self, predictions, label):
        loss = nn.CrossEntropyLoss()(predictions, label)
        return loss
    
    # To get the accuracy of 1 iteration
    def compute_accuracy(self, predictions, label):
        # Get the index of the maximum value in predictions
        predicted_label = torch.argmax(predictions)

        # Convert label to tensor
        label = torch.tensor(label).clone().detach().long()

        # Check if the prediction matches the true label
        correct_prediction = (predicted_label == label).item()

        # Calculate accuracy (1 if correct, 0 otherwise)
        accuracy = 1 if correct_prediction else 0

        return accuracy

    # Step function
    def train_step(self, target_embeddings, label, optimizer):
        # Set the model in training mode
        self.train()

        # Convert label to tensor
        label = torch.tensor(label).clone().detach().long()

        # Convert input to tensor
        target_embeddings = torch.tensor(target_embeddings, dtype=torch.float32)
        
        # Pass embeddings through the model
        predictions = self.forward(target_embeddings)

        # Calculate the loss
        loss = self.compute_loss(predictions, label)

        # Backpropagation and weight update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        accuracy = self.compute_accuracy(predictions, label)

        return loss.item(), accuracy

    # Training
    def train_model(self, optimizer, epochs, path):
        best_val_loss = float('inf')
        no_improvement_count = 0

        # Training loop
        start_time = time.time()
        for epoch in range(epochs):
            total_loss = 0.0
            total_accuracy = 0.0
            for input, label in zip(self.inputs, self.labels):
                loss, accuracy = self.train_step(input, label, optimizer)
                total_loss += loss
                total_accuracy += accuracy

            average_loss = total_loss / len(self.inputs)
            average_accuracy = total_accuracy / len(self.inputs)
            self.losses.append(average_loss)
            self.accuracies.append(average_accuracy)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {average_loss}, Accuracy: {average_accuracy}, Time: {time.time() - start_time}")

            if epoch % 20 == 0:
                if average_loss < best_val_loss:
                    best_val_loss = average_loss
                    no_improvement_count = 0
                else :
                    no_improvement_count += 1

            if epoch % 100 == 0 :
                save_path=os.path.join(path, 'model.pth')
                torch.save(self.state_dict(), save_path)
                print("Model saved successfully.")

            if average_accuracy == 1.0 or no_improvement_count >= PATIENCE:
                print("Training stopped. Accuracy reached 1.0 or Early Stopping.")
                save_path=os.path.join(path, 'model.pth')
                torch.save(self.state_dict(), save_path)
                print("Model saved successfully.")
                break

    # Plotting model's metrics
    def plot_metrics(self, path=None):
        # Plotting loss
        plt.figure()
        plt.plot(self.losses, label='Loss')
        plt.title('Evolution of Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        if path:
            plt.savefig(os.path.join(path, 'loss_plot.png'))
        else:
            plt.show()

        # Plotting accuracy
        plt.figure()
        plt.plot(self.accuracies, label='Accuracy')
        plt.title('Evolution of Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        if path:
            plt.savefig(os.path.join(path, 'accuracy_plot.png'))
        else:
            plt.show()

        # Close all figures
        plt.close('all')


def main():
    
    # Unmasked model
    unmasked_model = NN_1024_768_300_EMB(unmasked_dataset)
    unmasked_optimizer = optim.Adam(unmasked_model.parameters(), lr=LR)
    unmasked_model.train_model(unmasked_optimizer, epochs=NB_EPOCHS, path=UNMASKED_REID_MODEL_LOCATION)
    unmasked_model.plot_metrics(path=UNMASKED_REID_MODEL_LOCATION)
    
    # Masked models
    masked_with_unmasked_model = NN_1024_768_300_EMB(masked_with_unmasked_dataset)
    masked_with_unmasked_optimizer = optim.Adam(masked_with_unmasked_model.parameters(), lr=LR)
    masked_with_unmasked_model.train_model(masked_with_unmasked_optimizer, epochs=NB_EPOCHS, path=MASKED_WITH_UNMASKED_REID_MODEL_LOCATION)
    masked_with_unmasked_model.plot_metrics(path=MASKED_WITH_UNMASKED_REID_MODEL_LOCATION)
    
    masked_without_unmasked_model = NN_1024_768_300_EMB(masked_without_unmasked_dataset)
    masked_without_unmasked_optimizer = optim.Adam(masked_without_unmasked_model.parameters(), lr=LR)
    masked_without_unmasked_model.train_model(masked_without_unmasked_optimizer, epochs=NB_EPOCHS, path=MASKED_WITHOUT_UNMASKED_REID_MODEL_LOCATION)
    masked_without_unmasked_model.plot_metrics(path=MASKED_WITHOUT_UNMASKED_REID_MODEL_LOCATION)

    # Augmented Masked models
    augmented_masked_with_unmasked_model = NN_1024_768_300_EMB(augmented_masked_with_unmasked_dataset)
    augmented_masked_with_unmasked_optimizer = optim.Adam(augmented_masked_with_unmasked_model.parameters(), lr=LR)
    augmented_masked_with_unmasked_model.train_model(augmented_masked_with_unmasked_optimizer, epochs=NB_EPOCHS, path=AUGMENTED_MASKED_WITH_UNMASKED_REID_MODEL_LOCATION)
    augmented_masked_with_unmasked_model.plot_metrics(path=AUGMENTED_MASKED_WITH_UNMASKED_REID_MODEL_LOCATION)

    augmented_masked_without_unmasked_model = NN_1024_768_300_EMB(augmented_masked_without_unmasked_dataset)
    augmented_masked_without_unmasked_optimizer = optim.Adam(augmented_masked_without_unmasked_model.parameters(), lr=LR)
    augmented_masked_without_unmasked_model.train_model(augmented_masked_without_unmasked_optimizer, epochs=NB_EPOCHS, path=AUGMENTED_MASKED_WITHOUT_UNMASKED_REID_MODEL_LOCATION)
    augmented_masked_without_unmasked_model.plot_metrics(path=AUGMENTED_MASKED_WITHOUT_UNMASKED_REID_MODEL_LOCATION)

if __name__ == '__main__':
    main()
