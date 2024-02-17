import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the Excel file into a DataFrame
excel_file_path = 'Incipient motion.xlsx'  # Replace with the path to your Excel file
df = pd.read_excel(excel_file_path)
df = df.dropna()
X=df.iloc[:,1:-1]
y=df.iloc[:,-1]

# input_columns = df['g','y','Sf','G','d','v','u']
# output_column = df['tb']

# Create a boxplot using Seaborn
plt.figure(figsize=(10, 6))
sns.boxplot(data=X)

# Set labels and title
plt.xlabel('Parameters')
plt.ylabel('Values')
plt.title('Boxplot of Input and Output Parameters')

# Show the plot
plt.show()
