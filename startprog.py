import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 

url = 'diabetes.csv'
diabetes_data = pd.read_csv(url)
print(diabetes_data.head())

#data preprocessing
#finding the important columns with using correlation heatmap

correlation_matrix = diabetes_data.corr()
#create a mask to display only the lower triangle of the heatmap

#calculate the correlation matrix
correlation_matrix = diabetes_data.corr()

#create a mask to display only the lower triangle of the heatmap
mask = np.triu(np.ones_like(correlation_matrix, dtype = bool))
plt.figure(figsize= (10, 8))

#Generate the heatmap
sns.heatmap(correlation_matrix, annot = True, mask = mask, cmap = 'coolwarm', linewidths= 0.5, fmt = '.2f')
    
#Title for the heatmap
plt.title('correlation heatmap of diabetes dataset', fontsize = 16)

#Display the heatmap
plt.show()
