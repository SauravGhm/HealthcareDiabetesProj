import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlation_heatmap(data, output_path):
    """
    Plot a correlation heatmap for the dataset and save it to a file.
    """
    correlation_matrix = data.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f')
    
    plt.title('Correlation Heatmap')
    plt.savefig(output_path)
    plt.show()