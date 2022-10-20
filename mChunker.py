#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Get chunk Based similarity score.
"""
import pickle
import scipy
import torch
from torch import cuda
from sentence_transformers import models, SentenceTransformer
from transformers import BertTokenizerFast, BertConfig, BertForTokenClassification, BertTokenizer
from transformers import AutoModel, AutoTokenizer
import numpy as np
from numpy import dot
from numpy.linalg import norm
device = 'cuda' if cuda.is_available() else 'cpu'
print(device)

def getChunkMapping(chunks):
    i=0
    chunkids = []
    chunkids_map = {}
    for chunkid in range(len(chunks)):
        chunkwords = chunks[chunkid].split()
        for word in chunks[chunkid].split():
            chunkids.append(chunkid)
            chunkids_map[i] = chunkid
            i = i + 1
    return chunkids_map

def getChunkEmbedding(subwords_chunkids,output,totalchunks):
    chunkEmbeddings = []
    values = np.array(subwords_chunkids)
    for chunkid in range(totalchunks):
        indices = np.where(values == chunkid)[0]
        arr = output[indices[0]:indices[-1]+1]
        chunkemb = np.mean(np.array(arr),axis=0)
        chunkEmbeddings.append(chunkemb)
    return chunkEmbeddings

def getSubwords_chunkids(encoded_input,chunkids_map):
    subwords = encoded_input.word_ids()
    subwords_chunkids = []
    for i in range(1, len(subwords)-1):
        subwords_chunkids.append(chunkids_map[subwords[i]])
    #print(encoded_input)
    return subwords_chunkids

#using the finetuned chunker model, get labels (predicted) for every token in a sentence.
def getSentenceChunkLabels(sentence):
    inputs = tokenizer(sentence.split(),
                    is_split_into_words=True,
                    return_offsets_mapping=True, 
                    padding='max_length', 
                    truncation=True, 
                    max_length=128,
                    return_tensors="pt")
    
   # print(inputs)
    # move to gpu
    ids = inputs["input_ids"].to(device)
    mask = inputs["attention_mask"].to(device)
    # forward pass
    outputs = model(ids, attention_mask=mask)
    
    #attention = outputs[-1]
    logits = outputs[0]

    active_logits = logits.view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
    flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size*seq_len,) - predictions at the token level

    tokens = tokenizer.convert_ids_to_tokens(ids.squeeze().tolist())
   # print(tokens)
    token_predictions = [ids_to_labels[i] for i in flattened_predictions.cpu().numpy()]
    wp_preds = list(zip(tokens, token_predictions)) # list of tuples. Each tuple = (wordpiece, prediction)

    prediction = []
    for token_pred, mapping in zip(wp_preds, inputs["offset_mapping"].squeeze().tolist()):
      #only predictions on first word pieces are important
      if mapping[0] == 0 and mapping[1] != 0:
        prediction.append(token_pred[1])
      else:
        continue
    return prediction

#Based on the predicted labels for each token, this function returns list of chunks for a given sentence,prediction pair.
def getSentenceChunks(sentence,prediction):
    sentence_list = sentence.split()
    chunk_list = []
    for i in range(len(prediction)):
        #Identify the beginning of a chunk and iterates till beginning of next chunk is encountered.
        if('B' == prediction[i][0]):
            chunk = [] #initialize a new chunk
            chunk.append(sentence_list[i])
            j = i+1
            while(j<len(prediction) and 'B' != prediction[j][0]):
                chunk.append(sentence_list[j])
                j = j+1
            i = j
            chunk_list.append(" ".join(chunk))  #append chunk to the chunk_list 
    #return list of chunks
    return chunk_list


def getSentChunkWiseEmbedding(sentence,tokenizer,model):
    #get chunk labels for each token
    predicted_labels = getSentenceChunkLabels(sentence) 
    #divides sentences into chunks based on predicted labels
    list_of_chunks = getSentenceChunks(sentence,predicted_labels)
    #get chunk to word mapping
    chunk2Word_map = getChunkMapping(list_of_chunks)
    
    encoded_input = tokenizer(sentence.split(),
                              is_split_into_words = True, 
                              padding=True, 
                              truncation=True, 
                              max_length=128, 
                              return_tensors='pt')
    #map subwords to chunkids
    subwords_chunkid_map = getSubwords_chunkids(encoded_input,chunk2Word_map)
    
    with torch.no_grad():
         states = model(**encoded_input).hidden_states
    encoded_output = torch.stack([states[i] for i in range(len(states))])
    encoded_output = encoded_output.squeeze()
    #print("Output shape is {}".format(output.shape)) #7 x sen_len X 768
    encoded_output = encoded_output[-1] #Take the last hidden layer [sen_len X 768]
    
    #Get chunk embeddings by mean pooling all the subword embeddings belonging to one chunk id
    chunkEmbedding = getChunkEmbedding(subwords_chunkid_map,encoded_output[1:-1],len(list_of_chunks)) #removing cls and sep
               
            
    #print("list of chunks :", list_of_chunks, '\n', 'Embedding in progress!')
    #encode the sentence chunks as a list of chunk-wise embeddings.
    return chunkEmbedding,list_of_chunks

#For every chunk in sent2, n top similar chunks in sent1 are retrieved
def getChunkSimilarityScore(closest_n,sentencemeb1,sentencemeb2):
    total = 0
    for sentemeb2 in sentencemeb2:
        distances = scipy.spatial.distance.cdist([sentemeb2], sentencemeb1, "cosine")[0]

        results = zip(range(len(distances)), distances)
        results = sorted(results, key=lambda x: x[1])
        i=0
        for idx, distance in results[0:closest_n]:
            if(i==0):
                total = total + (1-distance)
     
    return total

def getSentenceChunkWiseEmbedding(sentence,sent_Emb_Model):
    #get chunk labels for each token
    predicted_labels = getSentenceChunkLabels(sentence) 
    #divides sentences into chunks based on predicted labels
    list_of_chunks = getSentenceChunks(sentence,predicted_labels)
    
    #print("list of chunks :", list_of_chunks, '\n', 'Embedding in progress!')
    #encode the sentence chunks as a list of chunk-wise embeddings.
    return sent_Emb_Model.encode(list_of_chunks),list_of_chunks

def evaluateSentencePair(src_sent,mt_sent):
    beta = 3
    sentence1 = src_sent
    sentence2 = mt_sent
    try:
        sentencemeb1,list_of_chunks1 = getSentChunkWiseEmbedding(sentence1,sentemb_tokenizer,sentemb_Model)
    except:
        sentencemeb1,list_of_chunks1 = getSentenceChunkWiseEmbedding(sentence1,sentEmbeddingModel)
        #print(sentence1,list_of_chunks1)
    len_chunk_1 = len(list_of_chunks1)
    try:
        sentencemeb2,list_of_chunks2 = getSentChunkWiseEmbedding(sentence2,sentemb_tokenizer,sentemb_Model)
    except:
        sentencemeb2,list_of_chunks2 = getSentenceChunkWiseEmbedding(sentence2,sentEmbeddingModel)
    len_chunk_2 = len(list_of_chunks2)   
    #print("Sentence 1 chunks : ",list_of_chunks1, "\n No of Chunks : ", len(list_of_chunks1))
    #print("Sentence 2 chunks : ",list_of_chunks2, "\n No of Chunks : ", len(list_of_chunks2))
    
    
    #For every chunk in mt, matching chunks in src are retrieved.
    Score2 = getChunkSimilarityScore(1,sentencemeb1,sentencemeb2) 
    try:
        Precision_Score = Score2/len_chunk_2
    except:
        Precision_Score = 0.01
    #For every chunk in src, matching chunks in mt are retrieved.
    Score1 = getChunkSimilarityScore(1,sentencemeb2,sentencemeb1) 
    try:
        Recall_Score = Score1/len_chunk_1
    except:
        Recall_Score = 0.01
    
    #compute parametrized f1 score
    beta = 3
    fbeta_score = (1+beta**2) * (Precision_Score*Recall_Score)/((beta**2) * Precision_Score + Recall_Score)

    return Precision_Score,Recall_Score,fbeta_score

def getSentenceEmbedding(model,sen_list):
    cos_sim = []
    embeddings = []
    #Compute sentence embeddings using LaBSE 
    embeddings = model.encode(sen_list)
    #Compute cosine-similarities
    for k in range(1,len(sen_list)):
        cos_sim = dot(embeddings[0], embeddings[k])/(norm(embeddings[0])*norm(embeddings[k]))  
    return cos_sim

def getChunkBasedScore(list1,list2):
    scores = []
    for i in range(len(list1)):
        try:
            p,r,s = evaluateSentencePair(list1[i],list2[i])
        except:
            s = 0.1
            
        Lb_sen_score = getSentenceEmbedding(sentEmbeddingModel,[list1[i],list2[i]])
        
        #print(s, Lb_sen_score)
        mean = (Lb_sen_score + s)/2
        scores.append(mean)
    return scores

#**********************************************************************************************************
#load tokenizer, model and ids_to_labels dict from finetuned Chunker Model
path_to_Chunker_Model = "ChunkerModel_BI"
tokenizer = BertTokenizerFast.from_pretrained(path_to_Chunker_Model)
model = BertForTokenClassification.from_pretrained(path_to_Chunker_Model,output_attentions=True)
model.to(device)
with open(path_to_Chunker_Model + "/idsToLabels_dict", 'rb') as handle:
    ids_to_labels = pickle.load(handle)
#**********************************************************************************************************
#load the tokenizer and model from pretrained sentence transformer model.
path = "sentence-transformers/distiluse-base-multilingual-cased"
sentemb_tokenizer = AutoTokenizer.from_pretrained(path)
sentemb_Model = AutoModel.from_pretrained(path,output_hidden_states=True)

#In case of chunk wise error,
sentEmbeddingModel = SentenceTransformer('sentence-transformers/LaBSE')

'''
ref = "On weekdays, the traffic jam stretches to the bridge over Albersloher Weg - and sometimes even goes over it."
mt = "An Wochentagen erstreckt sich die Verkehrsmarche auf die Brücke über Albersloh Weg - und manchmal auch über sie." 
score = getChunkBasedScore([ref],[mt])
print(score)
'''
