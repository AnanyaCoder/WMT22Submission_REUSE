# -*- coding: utf-8 -*-
"""
WMT 2022 MEtric Shared Task
Evaluating the MT outputs at Segment Level using MEE2, MEE4 metric.
@author: Ananya Mukherjee
"""

from gensim.models import KeyedVectors
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, util
import numpy as np
from numpy.linalg import norm
from numpy import dot
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
SF = SmoothingFunction()

def get_exact_match(ref1,hyp1):
    exact_matches=[]
    hypnew = []
    for word_hyp in hyp1:
        k=0
        #print('hypothesis',word_hyp)
        for word_ref in ref1:
            if(word_hyp==word_ref):
               # print(word_hyp,word_ref)
                exact_matches.append((word_hyp,word_ref))
                ref1.remove(word_ref)
                k=1
                break
            
        if(k==0):
            hypnew.append(word_hyp)
    return exact_matches,hypnew,ref1
	

def get_root_syn_match(ref,hyp,model):
    root_matches=[] #list to hold the words which satisfy root match
    syn_matches=[] #list to hold the words which satisfy synonym match
    #print(ref)
    for i in range(len(hyp)):
        #print('hypothesis', hyp[i])
        for j in range(len(ref)):
            #print('reference', ref[j])
            try:
                
                score = model.similarity(hyp[i],ref[j])
                #print(hyp[i],ref[j],score)
                if(score>=0.5):
                    root_matches.append((hyp[i],ref[j]))
                    #print("r",hyp[i],ref[j],score)
                    #if the word in hypothesis is matched with word in reference then remove that particular indexed word from reference
                    ref.remove(ref[j])
                    break
                    
                if(score>=0.4):
                    syn_matches.append((hyp[i],ref[j]))
                    #print("s",hyp[i],ref[j],score)
                    #if the word in hypothesis is matched with word in reference then remove that particular indexed word from reference
                    ref.remove(ref[j])
                    break
                
            except:
                #print('error (hyp,ref) ',hyp[i],ref[j])
                pass
    return root_matches,syn_matches
  
  
def get_fmean(alpha,matches_count,tl,rl):
    #matches_count = len(matches)
    try:
        precision = float(matches_count)/tl
        recall = float(matches_count)/rl
        fmean = (precision*recall)/(alpha*precision+(1-alpha)*recall)
    except ZeroDivisionError:
        return 0.0        
    return fmean

def get_MEE(ref,hyp,wordModel):
    #em = exact matches
    #rm = root matches
    #sm = synonym matches
    ref_sentence = ref
    hyp_sentence = hyp
    ref_len=len(ref_sentence)
    hyp_len=len(hyp_sentence)
    #print(ref_sentence,ref_len,hyp_sentence,hyp_len)
    em,unmatched_hyp,unmatched_ref = get_exact_match(ref_sentence,hyp_sentence)
    rm,sm = get_root_syn_match(ref_sentence,hyp_sentence,wordModel)
    exact_match = get_fmean(0.9,len(em),hyp_len,ref_len)
    root_match = get_fmean(0.9,len(em+rm),hyp_len,ref_len)
    syn_match = get_fmean(0.9,len(em+rm+sm),hyp_len,ref_len)
    m_score=(exact_match+root_match+syn_match)/3
    return m_score


def getSentenceEmbedding(model,sen_list):
    cos_sim = []
    embeddings = []
    #Compute sentence embeddings using LaBSE 
    embeddings = model.encode(sen_list)
    #Compute cosine-similarities
    for k in range(1,len(sen_list)):
        cos_sim = dot(embeddings[0], embeddings[k])/(norm(embeddings[0])*norm(embeddings[k]))  
    return cos_sim

def BleuMetric(ref,hyp):
    try:
        k = int(np.ceil(len(ref)/3))
        new_weights = [1 / k for _ in range(k)]
        bscore = sentence_bleu([ref],hyp,weights = new_weights,smoothing_function=SF.method4)
    except:
        return 0.0
    
    return bscore


def FastText_Emb(lngpair):
    model=KeyedVectors.load_word2vec_format('fasttext/cc.'+lngpair[3:]+ '.300.vec')
    #model=' '
    return model

def LaBSE_Emb():
    return SentenceTransformer('LaBSE')

def getModuleScore(mt,ref,lngpair,mee_version):
    score = []
    #Load fasttext Models before hand
    fasttextWordModel = FastText_Emb(lngpair)
    labseModel = LaBSE_Emb()
    
   
    for i in range(len(ref)):
        #Modified BLEU (ngram is based on the reference sentence length)
        modifiedBscore = BleuMetric(ref[i],mt[i])
        ref_sen = ref[i].split()
        hyp_sen = mt[i].split()
        #ExactMatch+RootMatch+SynMatch uses Fasttext word Embeddings
        mee = get_MEE(ref_sen,hyp_sen,fasttextWordModel)
        #Sentence Similarity which is assumed to capture context using LaBSE
        Lb_sen_score = getSentenceEmbedding(labseModel,[ref[i],mt[i]])
        
        MEE2 = ( ( ((2*mee)+modifiedBscore)/3) + Lb_sen_score)/2
        MEE4 = ( ( ((2*mee)+modifiedBscore)/3) + 3*Lb_sen_score)/4
        
        if(mee_version == MEE2):
            score.append(MEE2)
        else
            score.append(MEE4)  
    return score


