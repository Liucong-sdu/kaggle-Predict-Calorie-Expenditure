import pandas as pd 
"""df= pd.read_csv('train.csv')
df.describe().to_csv('train_summary.csv')"""
import featuretools as ft
'''初始化一个EntitySet对象'''
df=pd.read_csv('train.csv')
es=ft.EntitySet(id='id')
es = es.add_dataframe(dataframe_name='train', dataframe=df, index='id')
"""合成体脂率和bmi特征"""
def Body_Fat_Percentage(BFP):
    return 1.20 * BFP + 0.23 * 25 - 5.4
def BMI(BMI):
    return 1.20 * BMI + 0.23 * 25 - 5.4
"""通过新合成的数据，使用featuretools进行特征工程"""
def feature_engineering(df):
    df
