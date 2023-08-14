import json
import matplotlib.pyplot as plt

def plot_one_graph(model_name):
    # Read the contents of JSON file
    with open(f'json/history_{model_name}.json', 'r') as file:
        data = json.load(file)

    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(list(data['accuracy'].values()))
    plt.plot(list(data['val_accuracy'].values()))
    plt.title(f'{model_name} Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(list(data['loss'].values()))
    plt.plot(list(data['val_loss'].values()))
    plt.title(f'{model_name} Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.show()


def plot_graphs(model_names):
    plt.figure(figsize=(12, 5))
    model_names = model_names
    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    for model_name in model_names:
        with open(f'json/history_{model_name}.json', 'r') as file:
            data = json.load(file)
        plt.plot(list(data['accuracy'].values()), label=f'{model_name} Train')
        plt.plot(list(data['val_accuracy'].values()), label=f'{model_name} Validation')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True, color='white')  # Add white grid lines
    plt.gca().set_facecolor('#EAEAF2')  # Set background color to light gray

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    for model_name in model_names:
        with open(f'json/history_{model_name}.json', 'r') as file:
            data = json.load(file)
        plt.plot(list(data['loss'].values()), label=f'{model_name} Train')
        plt.plot(list(data['val_loss'].values()), label=f'{model_name} Validation')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True, color='white')  # Add white grid lines
    plt.gca().set_facecolor('#EAEAF2')  # Set background color to light gray

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    # Plot graphs for just one model
    #model_name = 'inceptionv3'  #CNN1, CNN2, VGG19, inceptionv3
    #plot_one_graphs(model_name)

    # Plot graphs for 4 different models
    model_names1 = ['CNN1', 'CNN2', 'VGG19', 'inceptionv3']
    model_names2 = ['aug_unmasked', 'aug_masked', 'unaug_unmasked', 'unaug_masked'] #CNN2
    plot_graphs(model_names2)
