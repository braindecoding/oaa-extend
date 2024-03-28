# First, you'll need to import pandas. Ensure you have it installed in your environment.
import pandas as pd

# Replace '/path/to/your/VAE.csv' with the actual path to your CSV file.
# For example: df = pd.read_csv('/mnt/data/VAE.csv')
df = pd.read_csv('V1_1x1_mse\VAE.csv')

# Calculate the average (mean) of the column values. This assumes your CSV has a single column of numerical values.
average_value = df.mean()[0]

# Print the average value
print(f"The average value is: {average_value}")
