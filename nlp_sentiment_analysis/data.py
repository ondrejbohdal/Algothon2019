import csv
from elasticsearch import Elasticsearch
import json
import pandas as pd

ELASTICSEARCH_HOST = 'dsa-stg-newsarchive-api.fr-nonprod.aws.thomsonreuters.com'
ELASTICSEARCH_API_KEY = 'REMOVED' # your personal key for Data Science Accelerator access to News on Elasticsearch

# Creating Elasticsearch connection through api gateway
es = Elasticsearch(
    host=ELASTICSEARCH_HOST,
    port=443,
    headers={'X-api-key': ELASTICSEARCH_API_KEY}, 
    use_ssl=True, 
    timeout=30
)

def show_stats(index,doc_type,field):
    '''Show basic statistics for a specific index'''
    res = es.search(index=index, doc_type=doc_type, body={
        "aggs" : {
            "date_stats" : { "stats" : { "field" : field } }
        }
    })
    print("Count: %d\nEarliest: %s\nLatest: %s"%(res['aggregations']['date_stats']['count'],res['aggregations']['date_stats']['min_as_string'],res['aggregations']['date_stats']['max_as_string']))


def news_elastic_to_df(res):
    '''takes a json object (result from elasticsearch query) 
    from news archive and returns a dataframe'''
    
    res_df = pd.DataFrame.from_dict([
            {'body': i['_source']['data']['body'], 
             'mimeType': i['_source']['data']['mimeType'],
             'firstCreated': i['_source']['data']['firstCreated'],
             'language': i['_source']['data']['language'],
             'altId': i['_source']['data']['altId'],
             'headline': i['_source']['data']['headline'],
             'takeSequence': i['_source']['data']['takeSequence'],
             'subjects': i['_source']['data']['subjects'],
             'audiences': i['_source']['data']['audiences'],
             'versionCreated': i['_source']['data']['versionCreated'], 
             'provider': i['_source']['data']['provider'],
             'instancesOf': i['_source']['data']['instancesOf'],
             'id': i['_source']['data']['id'],
             'urgency': i['_source']['data']['urgency']       
            } 
            for i in res['hits']['hits']
        ])
    return res_df


index_name = 'newsarchive'
res = es.search(index=index_name, doc_type='doc_newsarchive', body={"size":50, "query": {"match_all" : { }}})

# print(json.dumps(res, indent=4, sort_keys=True))

show_stats("newsarchive","doc_newsarchive","data.versionCreated")

import os
current_file = os.path.abspath(os.path.dirname(__file__)) #older/folder2/scripts_folder
print(current_file)


correct_tickers_df = pd.read_csv(os.path.join(current_file, 'correct_tickers.csv'), header=0)
correct_tickers_list = correct_tickers_df['ticker'].to_list()
print(correct_tickers_list)

ticker_to_name_location = os.path.join(current_file, 'ticker_to_name.csv')

ticker_to_name_df = pd.read_csv(ticker_to_name_location, header=0)
del ticker_to_name_df['Unnamed: 0']

d = {}
for i in range(ticker_to_name_df.shape[0]):
    d[ticker_to_name_df.iloc[i,0]] = ticker_to_name_df.iloc[i,1]

print(d["HAL"])

def generate_dataset_for_ticker(ticker):
    if ticker in d.keys():
        company_name = d[ticker]
        query = {
            "size": 10,
            "query": {
                "bool": {
                    "must": [
                        {
                            "match": {
                                "data.language": {
                                    "query": "en"
                                }
                            }
                        },
                        # {
                        #     "match": {
                        #         "data.urgency": {
                        #             "query": 1
                        #         }
                        #     }
                        # },
                        {
                            "range": {
                                "data.firstCreated": {
                                    "gte": "2016-11-01",
                                    "lte": "2018-12-31"
                                }
                            }
                        },
                        {
                            "query_string": {
                                "default_field": "data.headline",
                                "query": company_name
                            }
                        }
                    ]
                }
            }
        }

        res = es.search(index=index_name, doc_type='doc_newsarchive', body=query)

        df = news_elastic_to_df(res)
        df = df[['firstCreated', 'headline']]
        df['firstCreated'] = df['firstCreated'].apply(lambda x: x[0:10]) 
        print(df.columns)
        df = df.groupby(['firstCreated'])['headline'].apply(lambda x: ';; '.join(x)).reset_index()

        df['firstCreated'] = pd.to_datetime(df['firstCreated'], dayfirst=True)
        res = df.sort_values('firstCreated', ascending=False)
        df.to_csv(os.path.join(current_file, 'nlp_datasets/%s.csv'%ticker), index=True)

tickers_with_errors = []
for ticker in ['IAC', 'HK']:
    try:
        generate_dataset_for_ticker(ticker)
    except:
        tickers_with_errors.append(ticker)

print(tickers_with_errors)
        

