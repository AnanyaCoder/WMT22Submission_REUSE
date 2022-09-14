import argparse
import os

import pandas as pd
from sacrebleu import corpus_bleu, corpus_chrf, sentence_bleu, sentence_chrf
import numpy as np
from tqdm import tqdm
from typing import Dict, List
import ModulesScores_MEE as am

SRC_PATH = "metrics_inputs/txt/generaltest2022/sources/generaltest2022.{}.src.{}"
REFA_PATH = "metrics_inputs/txt/generaltest2022/references/generaltest2022.{}.ref.refA.{}"
REFB_PATH = "metrics_inputs/txt/generaltest2022/references/generaltest2022.{}.ref.refB.{}"
REFSPELL_PATH = "metrics_inputs/txt/generaltest2022/references/generaltest2022.{}.ref.refC.{}"
SYSTEM_FOLDER = "metrics_inputs/txt/generaltest2022/system_outputs/{}/"
METADATA_PATH = "metrics_inputs/txt/generaltest2022/metadata/{}.tsv"

MQM_LANGUAGE_PAIRS = [
    "zh-en", "en-de", "en-ru"
]

DOMAINS = ["conversation", "ecommerce", "news", "social"]
'''
LANGUAGE_PAIRS = [
     'zh-en',
     'en-de',
     'en-ru',
     'uk-en',
     'cs-en',
     'liv-en',
     'ru-sah',
     'cs-uk',
     'en-zh',
     'en-uk',
     'en-liv',
     'de-en',
     'fr-de',
     'en-hr',
     'de-fr',
     'en-ja',
     'en-cs',
     'uk-cs',
     'ja-en',
     'sah-ru',
     'ru-en'
]
'''

LANGUAGE_PAIRS = [
     'en-de'
]

#'en-de', 'en-ru', 'en-zh',  'de-en', 'ru-en'

CHALLENGE_SETS = [
    "challenge_ist-unbabel",
    "challenge_hw-tsc",
    "challenge_edinburgh-zurich",
    "challenge_dfki"
]

CHALLENGE_SETS_SRC = "metrics_inputs/txt/challengesets2022/sources/{}.{}.src.{}"
CHALLENGE_SETS_REF = "metrics_inputs/txt/challengesets2022/references/{}.{}.ref.1.{}"
CHALLENGE_SETS_SYSA = "metrics_inputs/txt/challengesets2022/system_outputs/{}/{}.{}.hyp.systemA.{}"
CHALLENGE_SETS_SYSB = "metrics_inputs/txt/challengesets2022/system_outputs/{}/{}.{}.hyp.systemB.{}"

'''
CHALLENGE_SETS_LPS = [
    'ar-en', 'ca-en', 'en-de', 'en-gl', 'en-lt', 'en-sr', 'en-tr', 'fr-es', 'fr-mr', 'ga-en', 'hi-en', 'id-en', 
    'pt-en', 'ru-en', 'sk-en', 'sr-en', 'zh-fr', 'af-fa', 'en-bg', 'en-ca', 'en-fr', 'en-id', 'en-it', 'en-ja', 
    'en-ko', 'en-no', 'en-ru', 'en-zh', 'es-de', 'es-ko', 'gl-en', 'hr-en', 'hu-en', 'ja-fr', 'ko-fr', 'pl-sk', 
    'sl-en', 'ur-en', 'zh-es', 'zh-ja', 'af-en', 'cs-en', 'en-hi', 'en-sl', 'ru-fr', 'ta-en', 'en-hu', 'en-sk', 
    'ru-es', 'zh-en', 'be-en', 'de-ko', 'de-ru', 'en-cs', 'en-fa', 'en-pt', 'en-uk', 'en-ur', 'fi-en', 'fr-ja', 
    'hr-lv', 'ja-de', 'ja-en', 'ja-es', 'ko-en', 'ko-es', 'lt-bg', 'zh-de', 'de-en', 'de-es', 'de-ja', 'en-et', 
    'en-nl', 'en-ro', 'en-vi', 'es-ja', 'fa-af', 'fa-en', 'fr-zh', 'he-en', 'he-sv', 'hi-ar', 'ja-zh', 'lv-en', 
    'lv-hr', 'nl-en', 'pl-en', 'pl-mr', 'sr-pt', 'sv-he', 'th-en', 'uk-en', 'vi-en', 'vi-hy', 'ar-fr', 'ca-es', 
    'de-fr', 'en-hy', 'et-en', 'fr-ru', 'it-en', 'ja-ko', 'ko-de', 'sk-pl', 'sw-en', 'tr-en', 'ar-hi', 'bg-en', 
    'bg-lt', 'de-zh', 'en-ar', 'en-da', 'en-el', 'en-lv', 'en-mr', 'en-sv', 'fr-de', 'fr-ko', 'hy-en', 'ko-ja', 
    'ko-zh', 'lt-en', 'ru-de', 'da-en', 'en-be', 'en-es', 'en-he', 'en-pl', 'en-ta', 'es-zh', 'fr-en', 'hy-vi', 
    'mr-en', 'no-en', 'pt-sr', 'ro-en', 'el-en', 'en-af', 'en-fi', 'en-hr', 'es-ca', 'es-en', 'es-fr', 'sv-en', 
    'wo-en', 'zh-ko'
]
'''

CHALLENGE_SETS_LPS = [
'en-de'
]
# 'en-de', 'en-zh', 'en-ru','de-en','zh-en','ru-en'
#TODO: Change the function below and add your metric in order to score the translations provided
# Segment-level scoring function
def segment_level_scoring(samples: Dict[str, List[str]], metric: str,lp):
    """ Function that takes source, translations and references along with a metric and returns
    segment level scores.
    
    :param samples: Dictionary with 'src', 'mt', 'ref' keys containing source sentences, translations and 
        references respectively.
    :param metric: String with the metric name. 
        If 'BLEU' runs sentence_bleu from sacrebleu. 
        If chrF runs chrF from sacrebleu    
    """
    
    
    if metric == "chrF":
        scores = run_sentence_chrf(samples["mt"], samples["ref"])
        #print(scores)
        
    elif metric == "BLEU":
        scores = run_sentence_bleu(samples["mt"], samples["ref"])
        
    elif metric == "random":
        scores = np.random.random(size = len(samples["ref"]))
    elif metric == "MEE2":
        scores = am.getModuleScore(samples["mt"], samples["ref"],lp,metric)
    elif metric == "MEE4":
        scores = am.getModuleScore(samples["mt"], samples["ref"],lp,metric)
            
    else:
        raise Exception(f"{metric} segment_scoring is not implemented!!")

    return scores


#TODO: Change the function below and add your metric in order to score the systems provided
# System-level scoring function
def system_level_scoring(samples: Dict[str, List[str]], metric: str, scores=List[float]):
    """ Function that takes source, translations and references along with a metric and returns
    system level scores.
    
    :param samples: Dictionary with 'src', 'mt', 'ref' keys containing source sentences, translations and 
        references respectively.
    :param metric: String with the metric name. 
        If 'BLEU' runs sentence_bleu from sacrebleu. 
        If chrF runs chrF from sacrebleu
    :param scores: List with segment level scores coming from the segment_level_scoring function.  
        Change this function if your metric DOES NOT use a simple average across segment level scores   
    """
    if metric == "chrF":
        system_score = corpus_chrf(samples["mt"], [samples["ref"]]).score

    elif metric == "BLEU":
        system_score = corpus_bleu(samples["mt"], [samples["ref"],]).score
            
    else:
        system_score = sum(scores)/len(scores)

    return system_score


def read_data(language_pair: str):
    src_lang, trg_lang = language_pair.split("-")
    sources = [s.strip() for s in open(SRC_PATH.format(language_pair, src_lang)).readlines()]
    referenceA = [s.strip() for s in open(REFA_PATH.format(language_pair, trg_lang)).readlines()]

    assert len(referenceA) == len(sources)
    references = {"refA": referenceA}

    if os.path.isfile(REFB_PATH.format(language_pair, trg_lang)):
        references["refB"] = [s.strip() for s in open(REFB_PATH.format(language_pair, trg_lang)).readlines()]
        assert len(references["refB"]) == len(sources)
        
    if os.path.isfile(REFSPELL_PATH.format(language_pair, trg_lang)):
        references["refC"] = [s.strip() for s in open(REFSPELL_PATH.format(language_pair, trg_lang)).readlines()]
        assert len(references["refC"]) == len(sources)
    
    lp_systems = [
        (SYSTEM_FOLDER.format(language_pair)+s,  s.split(".")[-2])
        for s in os.listdir(SYSTEM_FOLDER.format(lp)) if s.endswith(trg_lang)
    ]
    
    system_outputs = {}
    for system_path, system_name in lp_systems:
        if "ref" in system_name:
            continue
        system_outputs[system_name] = [s.strip() for s in open(system_path).readlines()]
        assert len(system_outputs[system_name]) == len(sources)

    if lp in MQM_LANGUAGE_PAIRS:
        metadata = [s.strip().split() for s in open(METADATA_PATH.format(language_pair)).readlines()]
        assert len(metadata) == len(sources)
    else:
        metadata = None
    
    return sources, references, system_outputs, metadata

def read_challenge_sets(language_pair: str):
    src_lang, trg_lang = language_pair.split("-")
    challenge_sets = {
        challenge_set:  {} for challenge_set in CHALLENGE_SETS
    }
    for challenge_set in CHALLENGE_SETS:
        if os.path.exists(CHALLENGE_SETS_SRC.format(challenge_set, language_pair, src_lang)):
            challenge_sets[challenge_set]["src"] = [s.strip() for s in open(CHALLENGE_SETS_SRC.format(challenge_set, language_pair, src_lang)).readlines()]
            challenge_sets[challenge_set]["refA"]  = [s.strip() for s in open(CHALLENGE_SETS_REF.format(challenge_set, language_pair, trg_lang)).readlines()]
            assert len(challenge_sets[challenge_set]["refA"]) == len(challenge_sets[challenge_set]["src"])
            lp_system_paths = [
                CHALLENGE_SETS_SYSA.format(language_pair, challenge_set, language_pair, trg_lang),
                CHALLENGE_SETS_SYSB.format(language_pair, challenge_set, language_pair, trg_lang),
            ]
            for system_path, system_name in zip(lp_system_paths, ["systemA", "systemB"]):
                challenge_sets[challenge_set][system_name] = [s.strip() for s in open(system_path).readlines()]
            assert len(challenge_sets[challenge_set]["systemA"]) == len(challenge_sets[challenge_set]["systemB"])
    return challenge_sets

def run_sentence_bleu(candidates: list, references: list) -> list:
    """ Runs sentence BLEU from Sacrebleu. """
    assert len(candidates) == len(references)
    bleu_scores = []
    for i in tqdm(range(len(candidates)), desc="Running BLEU..."):
        bleu_scores.append(sentence_bleu(candidates[i], [references[i]]).score)
    return bleu_scores


def run_sentence_chrf(candidates: list, references: list) -> list:
    """ Runs sentence chrF from Sacrebleu. """
    assert len(candidates) == len(references)
    chrf_scores = []
    for i in tqdm(range(len(candidates)), desc="Running chrF..."):
        chrf_scores.append(
            sentence_chrf(hypothesis=candidates[i], references=[references[i]]).score
        )
    return chrf_scores


def segment_scores(source, references, system_outputs, metadata, language_pair, metric_name, testset="generaltest2022"):
    segment_scores = []
    system_scores = []
    for ref in references:
        for hyp in system_outputs:
            print (f"Scoring {testset}-{language_pair} system {hyp} with {ref}:")
            samples = {"src": source, "mt": system_outputs[hyp], "ref": references[ref]}
            scores = segment_level_scoring(samples, metric_name,language_pair)
            assert len(scores) == len(references[ref])
            assert len(references[ref]) == len(system_outputs[hyp])
            assert len(system_outputs[hyp]) == len(source)
            
            # Save Segment Scores
            for i in range(len(source)):
                if metadata is not None:
                    domain = metadata[i][0]
                    document = metadata[i][1]
                else:
                    domain = "all"
                    document = "-"
                
                segment_scores.append({
                    "METRIC": metric_name,
                    "LANG-PAIR": language_pair,
                    "TESTSET": testset,
                    "DOMAIN": domain,
                    "DOCUMENT": document,
                    "REFERENCE": ref,
                    "SYSTEM_ID": hyp,
                    "SEGMENT_ID": i+1,
                    "SEGMENT_SCORE": scores[i]
                })
            
       
            # Compute and save System scores for all domains.
            system_score = system_level_scoring(samples, metric_name, scores)            
            system_scores.append({
                "METRIC": metric_name,
                "LANG-PAIR": language_pair,
                "TESTSET": testset,
                "DOMAIN": "all",
                "REFERENCE": ref,
                "SYSTEM_ID": hyp,
                "SYSTEM_LEVEL_SCORE": system_score
            })
        

            # Compute and save System scores for each domain.
            if metadata is not None:
                for domain in DOMAINS:
                    domain_idx = [i for i in range(len(metadata)) if metadata[i][0] == domain]
                    domain_src = [source[idx] for idx in domain_idx]
                    domain_ref = [references[ref][idx] for idx in domain_idx]
                    domain_hyp  = [system_outputs[hyp][idx] for idx in domain_idx]
                    domain_scores  = [scores[idx] for idx in domain_idx]
                    domain_samples = {"src": domain_src, "mt": domain_hyp, "ref": domain_ref}
                    system_score = system_level_scoring(domain_samples, metric_name, domain_scores)
                    system_scores.append({
                        "METRIC": metric_name,
                        "LANG-PAIR": language_pair,
                        "TESTSET": testset,
                        "DOMAIN": domain,
                        "REFERENCE": ref,
                        "SYSTEM_ID": hyp,
                        "SYSTEM_LEVEL_SCORE": system_score
                    })
        
                
        for alt_ref in references.keys():
            if ref != alt_ref:
                print (f"Scoring {testset}-{language_pair} system {alt_ref} with {ref}:")
                samples = {"src": source, "mt": references[alt_ref], "ref": references[ref]}
                # Compute and Save Segment Scores
                scores = segment_level_scoring(samples, metric_name,language_pair)
                for i in range(len(source)):
                    if metadata is not None:
                        domain = metadata[i][0]
                        document = metadata[i][1]
                    else:
                        domain = "all"
                        document = "-"
                    
                    segment_scores.append({
                        "METRIC": metric_name,
                        "LANG-PAIR": language_pair,
                        "TESTSET": testset,
                        "DOMAIN": domain,
                        "DOCUMENT": document,
                        "REFERENCE": ref,
                        "SYSTEM_ID": alt_ref,
                        "SEGMENT_ID": i+1,
                        "SEGMENT_SCORE": scores[i]
                    })
                
                # Compute and save System scores for all domains.
                system_score = system_level_scoring(samples, metric_name, scores)           
                system_scores.append({
                    "METRIC": metric_name,
                    "LANG-PAIR": language_pair,
                    "TESTSET": testset,
                    "DOMAIN": "all",
                    "REFERENCE": ref,
                    "SYSTEM_ID": alt_ref,
                    "SYSTEM_LEVEL_SCORE": system_score
                })
                
                # Compute and save System scores for each domain.
                if metadata is not None:
                    for domain in DOMAINS:
                        domain_idx = [i for i in range(len(metadata)) if metadata[i][0] == domain]
                        domain_src = [source[idx] for idx in domain_idx]
                        domain_ref = [references[ref][idx] for idx in domain_idx]
                        domain_hyp  = [references[alt_ref][idx] for idx in domain_idx]
                        domain_scores  = [scores[idx] for idx in domain_idx]
                        domain_samples = {"src": domain_src, "mt": domain_hyp, "ref": domain_ref}
                        system_score = system_level_scoring(domain_samples, metric_name, domain_scores)
                        system_scores.append({
                            "METRIC": metric_name,
                            "LANG-PAIR": language_pair,
                            "TESTSET": testset,
                            "DOMAIN": domain,
                            "REFERENCE": ref,
                            "SYSTEM_ID": alt_ref,
                            "SYSTEM_LEVEL_SCORE": system_score
                        })
        

    return pd.DataFrame(segment_scores), pd.DataFrame(system_scores)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scores Newstest2020 segments."
    )
    parser.add_argument(
        "--baseline",
        help="Metric to run.",
        type=str,
    )
    args = parser.parse_args()
    segment_data, system_data = [], []
  
    for lp in CHALLENGE_SETS_LPS:
        challenge_sets_data = read_challenge_sets(lp)
        for challenge_set in CHALLENGE_SETS:
            # For some language pairs we don't have challenge sets from all teams.
            if len(challenge_sets_data[challenge_set]) == 0:
                continue
            source = challenge_sets_data[challenge_set]["src"]
            references = {"refA": challenge_sets_data[challenge_set]["refA"]}
            system_outputs = {
                "systemA": challenge_sets_data[challenge_set]["systemA"], 
                "systemB": challenge_sets_data[challenge_set]["systemB"]
            }
            segments, systems = segment_scores(source, references, system_outputs, None, lp, args.baseline, testset=challenge_set)
            segment_data.append(segments)
            system_data.append(systems)
      
    for lp in LANGUAGE_PAIRS:
        source, references, system_outputs, metadata = read_data(lp)
        segments, systems = segment_scores(source, references, system_outputs, metadata, lp, args.baseline)
        segment_data.append(segments)
        system_data.append(systems)
        
   
    segment_data = pd.concat(segment_data, ignore_index=True)
    segment_data.to_csv("scores/{}.seg.score".format(args.baseline), index=False, header=False, sep="\t")
    
   
    system_data = pd.concat(system_data, ignore_index=True)
    system_data.to_csv("scores/{}.sys.score".format(args.baseline), index=False, header=False, sep="\t")
   
