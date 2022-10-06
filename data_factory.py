import os
import pandas as pd
from config import SQUAD_N_TEST, RACE_N_TEST, CONTEXT, ANSWER, QUESTION, SQUAD_TRAIN, SQUAD_DEV, RACE_TRAIN, RACE_DEV, SCIQ_DEV
import json

def preprocess_race(df):
    # rename cols for consistency
    df.rename(columns={'context': CONTEXT,'correct':ANSWER, 'question': QUESTION}, inplace=True)

    # subset to columns needed
    df = df[[CONTEXT, QUESTION, ANSWER]]

    # drop nulls
    df = df.dropna()
    return df

def preprocess_sciq(df):
    # rename cols for consistency
    df.rename(columns={'support': CONTEXT,'correct_answer':ANSWER, 'question': QUESTION}, inplace=True)
    # subset to columns needed
    df = df[[CONTEXT, QUESTION, ANSWER]]

    # drop nulls
    df = df.dropna()
    return df

def preprocess(df):
    # rename cols for consistency
    df.rename(columns={'context_para': CONTEXT,'answer_text':ANSWER, 'question': QUESTION}, inplace=True)

    # subset to columns needed
    df = df[[CONTEXT, QUESTION, ANSWER]]

    # drop nulls
    df = df.dropna()
    return df

class SQUAD():

    def train_test(self):
        train_test_df = pd.read_csv(SQUAD_TRAIN)
        train_test_df = preprocess(train_test_df)

        # split data
        test = train_test_df[:SQUAD_N_TEST]
        train = train_test_df[SQUAD_N_TEST:]
        return train, test

    def dev(self):
        dev = pd.read_csv(SQUAD_DEV)
        return preprocess(dev)


class RACE():

    def train_test(self):
        # read dataframe
        train_test_df = pd.read_csv(RACE_TRAIN)
        train_test_df = preprocess_race(train_test_df)

        # split data
        test = train_test_df[:RACE_N_TEST]
        train = train_test_df[RACE_N_TEST:]
        return train, test

    def dev(self):
        dev = pd.read_csv(RACE_DEV)
        return preprocess_race(dev)

class CS():
    def __init__(self):
        print("open file")
        # path_to_data=os.path.join('data', 'CS', 'sample.jsonl')
        # data_df = pd.read_json(path_to_data, lines=True)
        # print(data_df.head())

    def train_test(self):
        return None

    def dev(self):
        # path_to_data = os.path.join('data', 'CS', 'sample.jsonl')
        path_to_data = os.path.join('data', 'CS', 'sample.jsonl')
        data_df = pd.read_json(path_to_data, lines=True)
        df1=data_df[['abstract']]
        df2 = data_df[['body_text']]
        df2=df2.rename(columns={'body_text':'abstract'})
        # df3 = pd.merge(df1,df2, right_index=True, left_index=True)
        df3 = pd.concat([df1,df2])
        return df3
        # return None
class SCIQ():
    def __init__(self):
        print("sciq dataset")

    def train_test(self):
        return None

    def dev(self):
        dev = pd.read_csv(SCIQ_DEV)
        return preprocess_sciq(dev)

class pubMed():
    def __init__(self):
        print("open file")
        path_to_data=os.path.join('data', 'pubmed', 'pubmed22n0001.xml')
        df = pd.read_xml(path_to_data, xpath='.//Abstract')
        print(df.head())

    def train_test(self):
        return None

    def dev(self):
        return None

class DataFactory():
    def get_train_test(self, name):
        if name == 'squad':
            return SQUAD().train_test()
        if name == 'race':
            return RACE().train_test()
        if name == 'cs':
            return
        if name == 'pubmed':
            return


    def get_dev(self,name):
        if name == 'squad':
            return SQUAD().dev()
        if name == 'race':
            return RACE().dev()
        if name == 'cs':
            return CS().dev()
        if name == 'sciq':
            return SCIQ().dev()
        if name == 'pubmed':
            return pubMed().dev()