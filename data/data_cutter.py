# -*- coding: utf-8 -*-

'''
Created on 2017/8/23

@author: pangw
'''
import codecs
import pandas as pd

def cutter_str_label(dataset, label_index, str_label, output):
    df = pd.read_csv(dataset, header = 0, delimiter = "\t", quoting = 3,encoding = 'utf-8')
    texts = []
    for i in range(0,len(df)):        
        if df[label_index][i] == str_label:
            texts.append(df["TEXT"][i])
    with open(output, "w", encoding = "utf-8") as f:
        for ele in texts:
            f.write(ele + "\n")
            
def cutter_bool_label(dataset, label_index, output):
    df = pd.read_csv(dataset, header = 0, delimiter = "\t", quoting = 3,encoding = 'utf-8')    
    texts = []
    for i in range(0,len(df)):
        if df[label_index][i]:
            texts.append(df["TEXT"][i])
    with open(output, "w", encoding = "utf-8") as f:
        for ele in texts:
            f.write(ele + "\n")
    
        
if __name__ == "__main__":
    #cutted_rel = cutter_bool_label(r"D:\Desktop\mA_project\sentiment_analysis\data\dev.tsv",
    #                     "RELEVANCE", 
    #                     r"D:\Desktop\mA_project\sentiment_analysis\data\test\relevance\relevance.pos")
    cutted_doc_polarity = cutter_str_label(r"D:\Desktop\mA_project\sentiment_analysis\data\dev.tsv",
                         "SENTIMENT",'neutral', 
                         r"D:\Desktop\mA_project\sentiment_analysis\data\test\sentiment\sentiment.neu")
    
    # cutted_asp_polarity = cutter(r"D:\Desktop\mA_project\sentiment_analysis\data\train.tsv",
    #                     "CATEGORY:SENTIMENT",'positive', 
    #                    r"D:\Desktop\mA_project\sentiment_analysis\data\train\sentiment\sentiment.pos")
    print('Cutting...')
    