import pandas as pd
df= pd.read_csv('train.csv')
df.describe().to_csv('train_summary.csv')