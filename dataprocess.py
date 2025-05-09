import pandas as pd 
"""df= pd.read_csv('train.csv')
df.describe().to_csv('train_summary.csv')"""
import featuretools as ft
'''初始化一个EntitySet对象'''
df1=pd.read_csv('train.csv')
df2=pd.read_csv('test.csv')
es1=ft.EntitySet(id='id')
es2=ft.EntitySet(id='id')
"""将数据集添加到EntitySet中"""
es1 = es1.add_dataframe(dataframe_name='train', dataframe=df1, index='id')
es2 = es2.add_dataframe(dataframe_name='test', dataframe=df2, index='id')
"""合成体脂率和bmi特征"""
def Body_Fat_Percentage(df):
    return 1.20 * df+ 0.23 * 25 - 5.4
def BMI(BMI):
    return 1.20 * BMI + 0.23 * 25 - 5.4
"""通过新合成的数据，使用featuretools进行特征工程并且编码"""
def feature_engineering(df):
    df
'''存储为新的csv文件'''

