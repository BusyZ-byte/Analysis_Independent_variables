import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats

# Load the data
df = pd.read_csv('data_ses_woa_by_postcode.csv', header=[0, 1], delimiter=';')

# Flatten the multi-index header
df.columns = [' '.join(col).strip() for col in df.columns.values]

# Rename the first column to 'Postcode'
df.rename(columns={df.columns[0]: 'Postcode'}, inplace=True)

# Convert columns to numeric
for col in df.columns[1:]:
 df[col] = pd.to_numeric(df[col], errors='coerce')

# Fill NaNs with 0
df = df.fillna(0)

## the rows with NaN values are exchanged with 0 because they are apparently industrial areas with mostly warehouses, office spaces & businesess

# Calculate Z-scores
columns_to_check = df.columns[1:]
z_scores = np.abs(stats.zscore(df[columns_to_check]))

# Define a threshold
threshold = 3
outliers = np.where(z_scores > threshold)

# Print outliers
print(f"Outliers are at index positions: {outliers}")


# Visualization (only for the first 3 data columns)
plt.figure(figsize=(10, 6))
plt.plot(df['Postcode'], df[df.columns[1]], label=df.columns[1])
plt.plot(df['Postcode'], df[df.columns[2]], label=df.columns[2])
plt.plot(df['Postcode'], df[df.columns[3]], label=df.columns[3])
plt.scatter(df.iloc[outliers[0]]['Postcode'], df.iloc[outliers[0]][df.columns[1]], color='r', label='Outliers')
plt.legend(loc='upper left')
plt.xlabel('Postcode')  # Add x-axis label
plt.ylabel('Score')  # Add y-axis label
plt.title('Anomaly detection')
plt.show()



# visualization for the next three columns
plt.figure(figsize=(10, 6))
plt.plot(df['Postcode'], df[df.columns[4]], label=df.columns[4])
plt.plot(df['Postcode'], df[df.columns[5]], label=df.columns[5])
plt.plot(df['Postcode'], df[df.columns[6]], label=df.columns[6])
plt.scatter(df.iloc[outliers[0]]['Postcode'], df.iloc[outliers[0]][df.columns[1]], color='r', label='Outliers')
plt.legend(loc='upper left')
plt.xlabel('Postcode')  # Add x-axis label
plt.ylabel('Score')  # Add y-axis label
plt.title('Anomaly detection')
plt.show()


# visualization for the next three columns
plt.figure(figsize=(10, 6))
plt.plot(df['Postcode'], df[df.columns[7]], label=df.columns[7])
plt.plot(df['Postcode'], df[df.columns[8]], label=df.columns[8])
plt.plot(df['Postcode'], df[df.columns[9]], label=df.columns[9])
plt.scatter(df.iloc[outliers[0]]['Postcode'], df.iloc[outliers[0]][df.columns[1]], color='r', label='Outliers')
plt.legend(loc='upper left')
plt.xlabel('Postcode')  # Add x-axis label
plt.ylabel('Score')  # Add y-axis label
plt.title('Anomaly detection')
plt.show()


#visualization for the next three columns
plt.figure(figsize=(10, 6))
plt.plot(df['Postcode'], df[df.columns[10]], label=df.columns[10])
plt.plot(df['Postcode'], df[df.columns[11]], label=df.columns[11])
plt.plot(df['Postcode'], df[df.columns[12]], label=df.columns[12])
plt.scatter(df.iloc[outliers[0]]['Postcode'], df.iloc[outliers[0]][df.columns[1]], color='r', label='Outliers')
plt.legend(loc='upper left')
plt.xlabel('Postcode')  # Add x-axis label
plt.ylabel('Score')  # Add y-axis label
plt.title('Anomaly detection')
plt.show()

#visualization for the next three columns
plt.figure(figsize=(10, 6))
plt.plot(df['Postcode'], df[df.columns[13]], label=df.columns[13])
plt.plot(df['Postcode'], df[df.columns[14]], label=df.columns[14])
plt.plot(df['Postcode'], df[df.columns[15]], label=df.columns[15])
plt.scatter(df.iloc[outliers[0]]['Postcode'], df.iloc[outliers[0]][df.columns[1]], color='r', label='Outliers')
plt.legend(loc='upper left')
plt.xlabel('Postcode')  # Add x-axis label
plt.ylabel('Score')  # Add y-axis label
plt.title('Anomaly detection')
plt.show()

#visualization for the next three columns
plt.figure(figsize=(10, 6))
plt.plot(df['Postcode'], df[df.columns[16]], label=df.columns[16])
plt.plot(df['Postcode'], df[df.columns[17]], label=df.columns[17])
plt.plot(df['Postcode'], df[df.columns[18]], label=df.columns[18])
plt.scatter(df.iloc[outliers[0]]['Postcode'], df.iloc[outliers[0]][df.columns[1]], color='r', label='Outliers')
plt.legend(loc='upper left')
plt.xlabel('Postcode')  # Add x-axis label
plt.ylabel('Score')  # Add y-axis label
plt.title('Anomaly detection')
plt.show()

#visualization for the next three columns
plt.figure(figsize=(10, 6))
plt.plot(df['Postcode'], df[df.columns[19]], label=df.columns[19])
plt.plot(df['Postcode'], df[df.columns[20]], label=df.columns[20])
plt.plot(df['Postcode'], df[df.columns[21]], label=df.columns[21])
plt.scatter(df.iloc[outliers[0]]['Postcode'], df.iloc[outliers[0]][df.columns[1]], color='r', label='Outliers')
plt.legend(loc='upper left')
plt.xlabel('Postcode')  # Add x-axis label
plt.ylabel('Score')  # Add y-axis label
plt.title('Anomaly detection')
plt.show()

#visualization for the next three columns
plt.figure(figsize=(10, 6))
plt.plot(df['Postcode'], df[df.columns[22]], label=df.columns[22])
plt.plot(df['Postcode'], df[df.columns[23]], label=df.columns[23])
plt.plot(df['Postcode'], df[df.columns[24]], label=df.columns[24])
plt.scatter(df.iloc[outliers[0]]['Postcode'], df.iloc[outliers[0]][df.columns[1]], color='r', label='Outliers')
plt.legend(loc='upper left')
plt.xlabel('Postcode')  # Add x-axis label
plt.ylabel('Score')  # Add y-axis label
plt.title('Anomaly detection')
plt.show()

#visualization for the next three columns
plt.figure(figsize=(10, 6))
plt.plot(df['Postcode'], df[df.columns[25]], label=df.columns[25])
plt.plot(df['Postcode'], df[df.columns[26]], label=df.columns[26])
plt.plot(df['Postcode'], df[df.columns[27]], label=df.columns[27])
plt.scatter(df.iloc[outliers[0]]['Postcode'], df.iloc[outliers[0]][df.columns[1]], color='r', label='Outliers')
plt.legend(loc='upper left')
plt.xlabel('Postcode')  # Add x-axis label
plt.ylabel('Score')  # Add y-axis label
plt.title('Anomaly detection')
plt.show()

#visualization for the next three columns
plt.figure(figsize=(10, 6))
plt.plot(df['Postcode'], df[df.columns[28]], label=df.columns[28])
plt.plot(df['Postcode'], df[df.columns[29]], label=df.columns[29])
plt.plot(df['Postcode'], df[df.columns[30]], label=df.columns[30])
plt.scatter(df.iloc[outliers[0]]['Postcode'], df.iloc[outliers[0]][df.columns[1]], color='r', label='Outliers')
plt.legend(loc='upper left')
plt.xlabel('Postcode')  # Add x-axis label
plt.ylabel('Score')  # Add y-axis label
plt.title('Anomaly detection')
plt.show()


# Define a threshold
threshold = 3
outliers_row_col_indices = np.where(z_scores > threshold)

# Create a DataFrame for outliers
outliers_df = pd.DataFrame(columns=df.columns)
for i in range(len(outliers_row_col_indices[0])):
    row_index = outliers_row_col_indices[0][i]
    col_index = outliers_row_col_indices[1][i]
    outliers_df = pd.concat([outliers_df, df.iloc[[row_index], :]])

import pandas as pd

# Set display options
pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', None)  # or 199


# Print outliers DataFrame
print(outliers_df)

# save outliers_df in a csv file
outliers_df.to_csv('outliers_ses_woa.csv', index=False)

