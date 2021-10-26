import numpy as np
import pandas as pd

def data_processing(path, iiAutorec = True, dat_file = False):
    """
    preprocessing data
    path: path of data file
    dat_file: file ".dat"
    """  
    def read_data(path):

        if dat_file:
            delimeter = "::"
        else:
            delimeter = None
        df = pd.read_csv(path, delimiter=delimeter)
        df.columns = ["userId", "movieId", "rating", "timestamps"]
        return df

    def train_test_split(df, frac=0.8, random_state = 13):
        trainset_tmp = df.groupby(df.userId)[['movieId', 'rating','timestamps']].apply(lambda x: x.sample(frac= frac, random_state = random_state)).reset_index()
        trainset = trainset_tmp.drop(labels='level_1', axis=1)
        testset_tmp = pd.merge(df,trainset, on= ['movieId', 'rating','timestamps'],how="outer",indicator=True)
        testset_tmp = testset_tmp[testset_tmp['_merge']=='left_only']
        testset = testset_tmp.drop(labels=['userId_y','_merge'], axis=1).rename(columns={'userId_x':'userId'})

        return trainset, testset

    def reset_idx(df, df_map):
        df_new = df_map.copy()
        user = df.userId.unique().tolist()
        movie = df.sort_values(by="movieId").movieId.unique().tolist()
        user_dict = dict(zip(user, list(range(len(user)))))
        movie_dict = dict(zip(movie, list(range(len(movie)))))
        df_new['userId'] = df_map['userId'].map(user_dict)
        df_new['movieId'] = df_map['movieId'].map(movie_dict)

        return df_new  

    df = read_data(path)
    df_train, df_test = train_test_split(df)
    train = reset_idx(df,df_train).values
    print(train)
    test = reset_idx(df,df_test).values
    num_user = len(df.userId.unique())
    num_movie = len(df.movieId.unique())
    if iiAutorec:
        trainset = np.zeros((num_movie,num_user),dtype='float32')
        testset = np.zeros((num_movie,num_user),dtype='float32')
        for row in train:
            trainset[int(row[1]),int(row[0])] = row[2]       
        for row in test:
            testset[int(row[1]),int(row[0])] = row[2] 
    else:
        trainset = np.zeros((num_user,num_movie),dtype='float32')
        testset = np.zeros((num_user,num_movie),dtype='float32')
        for row in train:
            trainset[int(row[0]),int(row[1])] = row[2]
        for row in train:
            testset[int(row[0]),int(row[1])] = row[2]      

    return trainset, testset