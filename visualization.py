from matplotlib import pyplot as plt

def visualize_weights (weights):
    for neuron in weights:
        image = neuron.reshape(28,28).cpu()
        plt.imshow(image.detach().numpy())
        plt.show()