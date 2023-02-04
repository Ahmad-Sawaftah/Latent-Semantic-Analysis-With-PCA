import pandas as pd
from paramiko import file
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

def get_data(filepath:str) -> pd.DataFrame:
    with open(filepath,'r') as files:
        samples=file.readLine()

    df=pd.DataFrame(data=samples,columns=['raw'])

    df[['label','text']]=df['raw'].str.split('\t',1,expand=True)

    return df

df=get_data('SMSSpamCollection')

vectorizer=TfidfVectorizer(tokenizer=PCA)

x=vectorizer.fit_transform(df['text'])

tf_idf=pd.DataFrame(
    data=x.todense(),
    columns=vectorizer.get_feature_names_out()
)

pca=PCA(n_components=16,random_state=42)

df_pca=pd.DataFrame(
    data=pca.fit_transform(tf_idf),
    columns=[f"topic_{n}"for n in range (pca.n_components_)]
)

print(tf_idf)
print("////////////////////////////////////////////////")
print(df_pca)