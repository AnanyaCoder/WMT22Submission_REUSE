import argparse
import os

import pandas as pd
from typing import Dict, List
import mChunker as m

SRC_PATH = "metrics_inputs/txt/generaltest2022/sources/generaltest2022.{}.src.{}"
REFA_PATH = "metrics_inputs/txt/generaltest2022/references/generaltest2022.{}.ref.refA.{}"
REFB_PATH = "metrics_inputs/txt/generaltest2022/references/generaltest2022.{}.ref.refB.{}"
REFC_PATH = "metrics_inputs/txt/generaltest2022/references/generaltest2022.{}.ref.refC.{}"
REFSTUD_PATH = "metrics_inputs/txt/generaltest2022/references/generaltest2022.{}.ref.refstud.{}"
SYSTEM_FOLDER = "metrics_inputs/txt/generaltest2022/system_outputs/{}/"
METADATA_PATH = "metrics_inputs/txt/generaltest2022/metadata/{}.tsv"

MQM_LANGUAGE_PAIRS = [
    "zh-en", "en-de", "en-ru"
]

DOMAINS = ["conversation", "ecommerce", "news", "social"]

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

CHALLENGE_SETS = [
    "challenge_ist-unbabel",
    "challenge_hw-tsc",
    "challenge_edinburgh-zurich",
    "challenge_edinburgh-zurich-v2",
    "challenge_dfki"
]

CHALLENGE_SETS_SRC = "metrics_inputs/txt/challengesets2022/sources/{}.{}.src.{}"
CHALLENGE_SETS_REF = "metrics_inputs/txt/challengesets2022/references/{}.{}.ref.1.{}"
CHALLENGE_SETS_SYSA = "metrics_inputs/txt/challengesets2022/system_outputs/{}/{}.{}.hyp.systemA.{}"
CHALLENGE_SETS_SYSB = "metrics_inputs/txt/challengesets2022/system_outputs/{}/{}.{}.hyp.systemB.{}"

CHALLENGE_SETS_LPS = [
    "zh-en","en-de","en-ru"
]

# Loading model just once.
#from comet import download_model, load_from_checkpoint

#model_path = download_model("wmt21-comet-qe-mqm")
#model = load_from_checkpoint(model_path)


#TODO: Change the function below and add your metric in order to score the translations provided
# Segment-level scoring function
def segment_level_scoring(samples: Dict[str, List[str]], metric: str):
    """ Function that takes source, translations and references along with a metric and returns
    segment level scores.
    
    :param samples: Dictionary with 'src', 'mt', 'ref' keys containing source sentences, translations and 
        references respectively.
    :param metric: String with the metric name. 
        If 'BLEU' runs sentence_bleu from sacrebleu. 
        If chrF runs chrF from sacrebleu    
    """
    if metric == "COMET-QE":
        data = [{"src": s, "mt": m} for s, m in zip(samples["src"], samples["mt"])]
        scores, _ = model.predict(data, batch_size=8, gpus=1)
    elif metric == "REUSE":
    	scores = m.getChunkBasedScore(samples["src"], samples["mt"])
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
    return sum(scores)/len(scores)


def read_data(language_pair: str):
    src_lang, trg_lang = language_pair.split("-")
    sources = [s.strip() for s in open(SRC_PATH.format(language_pair, src_lang)).readlines()]
    references = {}

    if os.path.isfile(REFA_PATH.format(language_pair, trg_lang)):
        references["refA"] = [s.strip() for s in open(REFA_PATH.format(language_pair, trg_lang)).readlines()]
        assert len(references["refA"]) == len(sources)

    if os.path.isfile(REFB_PATH.format(language_pair, trg_lang)):
        references["refB"] = [s.strip() for s in open(REFB_PATH.format(language_pair, trg_lang)).readlines()]
        assert len(references["refB"]) == len(sources)
        
    if os.path.isfile(REFC_PATH.format(language_pair, trg_lang)):
        references["refC"] = [s.strip() for s in open(REFC_PATH.format(language_pair, trg_lang)).readlines()]
        assert len(references["refC"]) == len(sources)
    
    if os.path.isfile(REFSTUD_PATH.format(language_pair, trg_lang)):
        references["refstud"] = [s.strip() for s in open(REFSTUD_PATH.format(language_pair, trg_lang)).readlines()]
        assert len(references["refstud"]) == len(sources)
    
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

def segment_scores(source, references, system_outputs, metadata, language_pair, metric_name, testset="generaltest2022"):
    segment_scores = []
    system_scores = []
    
    for hyp in system_outputs:
        print (f"Scoring {testset}-{language_pair} system {hyp} with src:")
        samples = {"src": source, "mt": system_outputs[hyp]}
        scores = segment_level_scoring(samples, metric_name)
        assert len(scores) == len(system_outputs[hyp])
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
                "REFERENCE": "src",
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
            "REFERENCE": "src",
            "SYSTEM_ID": hyp,
            "SYSTEM_LEVEL_SCORE": system_score
        })

        # Compute and save System scores for each domain.
        if metadata is not None:
            for domain in DOMAINS:
                domain_idx = [i for i in range(len(metadata)) if metadata[i][0] == domain]
                domain_src = [source[idx] for idx in domain_idx]
                domain_hyp  = [system_outputs[hyp][idx] for idx in domain_idx]
                domain_scores  = [scores[idx] for idx in domain_idx]
                domain_samples = {"src": domain_src, "mt": domain_hyp}
                system_score = system_level_scoring(domain_samples, metric_name, domain_scores)
                system_scores.append({
                    "METRIC": metric_name,
                    "LANG-PAIR": language_pair,
                    "TESTSET": testset,
                    "DOMAIN": domain,
                    "REFERENCE": "src",
                    "SYSTEM_ID": hyp,
                    "SYSTEM_LEVEL_SCORE": system_score
                })
                
    for ref in references.keys():
        print (f"Scoring {testset}-{language_pair} system {ref} with src:")
        samples = {"src": source, "mt": references[ref]}

        # Compute and Save Segment Scores
        scores = segment_level_scoring(samples, metric_name)
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
                "REFERENCE": "src",
                "SYSTEM_ID": ref,
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
            "REFERENCE": "src",
            "SYSTEM_ID": ref,
            "SYSTEM_LEVEL_SCORE": system_score
        })
                
        # Compute and save System scores for each domain.
        if metadata is not None:
            for domain in DOMAINS:
                domain_idx = [i for i in range(len(metadata)) if metadata[i][0] == domain]
                domain_src = [source[idx] for idx in domain_idx]
                domain_hyp  = [references[ref][idx] for idx in domain_idx]
                domain_scores  = [scores[idx] for idx in domain_idx]
                domain_samples = {"src": domain_src, "mt": domain_hyp}
                system_score = system_level_scoring(domain_samples, metric_name, domain_scores)
                system_scores.append({
                    "METRIC": metric_name,
                    "LANG-PAIR": language_pair,
                    "TESTSET": testset,
                    "DOMAIN": domain,
                    "REFERENCE": "src",
                    "SYSTEM_ID": ref,
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
    segment_data.to_csv("scores/qe-as-a-metric/{}.seg.score".format(args.baseline), index=False, header=False, sep="\t")
    
    system_data = pd.concat(system_data, ignore_index=True)
    system_data.to_csv("scores/qe-as-a-metric/{}.sys.score".format(args.baseline), index=False, header=False, sep="\t")
