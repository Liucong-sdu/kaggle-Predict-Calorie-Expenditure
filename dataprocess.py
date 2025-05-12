from sklearn.preprocessing import OneHotEncoder
import pandas as pd 
import featuretools as ft

# 加载数据
df1 = pd.read_csv('train.csv')
df2 = pd.read_csv('test.csv')

# 独热编码（处理未知类别）
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_train = encoder.fit_transform(df1[['Sex']])
encoded_test = encoder.transform(df2[['Sex']])

# 合并编码结果
encoded_df_train = pd.DataFrame(encoded_train, columns=encoder.get_feature_names_out(['Sex']))
encoded_df_test = pd.DataFrame(encoded_test, columns=encoder.get_feature_names_out(['Sex']))
df1 = pd.concat([df1, encoded_df_train], axis=1).drop('Sex', axis=1)
df2 = pd.concat([df2, encoded_df_test], axis=1).drop('Sex', axis=1)

# 合成BMI和体脂率
def BMI(df):
    df['BMI'] = df['Weight'] / ((df['Height']/100 )**2)
    return df

def Body_Fat_Percentage(df):
    df['Body_Fat_Percentage'] = 1.20 * df['BMI'] + 0.23 * df['Age'] - 5.4 - 10.8*df['Sex_male']
    return df

df1 = BMI(df1)
df2 = BMI(df2)
df1 = Body_Fat_Percentage(df1)
df2 = Body_Fat_Percentage(df2)

df1.to_csv('train_processed.csv', index=False)
df2.to_csv('test_processed.csv', index=False)
# 初始化Woodwork Schema
"""df1.ww.init(
    name='train',
    index='id',
    logical_types={
        # 原始数值列
        'Age': 'Double',
        'Height': 'Double',
        'Weight': 'Double',
        'Duration': 'Double',
        'Heart_Rate': 'Double',
        'Body_Temp': 'Double',
        'Calories': 'Double',
        # 独热编码后的分类列
        'Sex_female': 'Categorical',
        'Sex_male': 'Categorical',
        # 合成的新列
        'BMI': 'Double',
        'Body_Fat_Percentage': 'Double'
    }
)
df2.ww.init(
    name='test',
    index='id',
    logical_types={
        # 原始数值列
        'Age': 'Double',
        'Height': 'Double',
        'Weight': 'Double',
        'Duration': 'Double',
        'Heart_Rate': 'Double',
        'Body_Temp': 'Double',
        # 独热编码后的分类列
        'Sex_female': 'Categorical',
        'Sex_male': 'Categorical',
        # 合成的新列
        'BMI': 'Double',
        'Body_Fat_Percentage': 'Double'
    }
)
# 创建EntitySet并添加DataFrame
es1 = ft.EntitySet(id='train_es')
es1 = es1.add_dataframe(
    dataframe_name='train',
    dataframe=df1,
    
    logical_types=df1.ww.logical_types
)

es2 = ft.EntitySet(id='test_es')
es2 = es2.add_dataframe(
    dataframe_name='test',
    dataframe=df2,
    
    logical_types=df2.ww.logical_types
)

# 生成特征
features1, feature_defs1 = ft.dfs(
    entityset=es1,
    target_dataframe_name='train',  # 与DataFrame名称一致
    verbose=True,
    n_jobs=1,
    max_depth=100
)

features2, feature_defs2 = ft.dfs(
    entityset=es2,
    target_dataframe_name='test',
    verbose=True,
    n_jobs=1,

    max_depth=100
)

# 保存结果
features1.to_csv('train_features.csv', index=False)
features2.to_csv('test_features.csv', index=False)
ft.save_features(feature_defs1, 'train_feature_defs.json')
ft.save_features(feature_defs2, 'test_feature_defs.json')"""



