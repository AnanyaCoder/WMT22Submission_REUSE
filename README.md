# WMT22Submission_REUSE
## README

This contains the inputs for the metrics task at WMT 2022, both the main task ( from WMT general translation task) and the challenge sets subtask (created specifically to test MT metrics). Please score both. You can use the provided prepare_scores.py script to iterate over all files. 

The files can be found in two formats:
- a text format, where one line corresponds to one segment.  
- an xml format, for those who would like to make use of document boundaries.  https://github.com/wmt-conference/wmt-format-tools/tree/main/wmtformat contains some potentially useful scripts to process the xml files. Note that we dont have document boundaries for the challenge sets, and you'll need to score these from the txt subfolder.

The txt dir contains one subdir for generaltest2022 and challengesets2022, each of which contains:
- sources (dir with all the source files, filenames look like <testsetname>.<src>-<tgt>.ref.<refid>.<tgt>)
- references (dir with all the ref files, filenames look like <testsetname>.<src>-<tgt>.src.<src>)
- system_outputs
   |_ <src>-<tgt> (filenames look like  <testsetname>.<src>-<tgt>.hyp.<systemid>.<tgt>) 
   
In addition, the generaltest2022 contains a metadata subdir with information of domain and document id for each segment.



# Download and extract the Chunker Model from the below link.

Chunker Model Link : https://iiitaphyd-my.sharepoint.com/:u:/g/personal/ananya_mukherjee_research_iiit_ac_in/Ede2Yu2U9ZBHsD2Yu1PRiZ4BkuDx2GED3E-gXBCaGMlN1Q?e=DvzH8k

# Scoring Scripts:

Along with the inputs we are releasing a script to help everyone score the data in the correct format.

To adapt that to your metric you just need to change the segment_level_scoring function and the system_level_scoring according to your metric. These methods receive a dictionary called samples containing the following format:

samples = {
    "src": ["hello world!", "welcome to Metrics task", ...],
    "mt": ["ola mundo!", "bem vindos à tarefa de metricas", ...],
}

For QE-as-a-metric we provide a similar script to run COMET-QE baseline (first verify if you have unbabel-comet installed!):

```bash
python3 prepare_qe_scores_REUSE.py --baseline COMET-QE
```
For REUSE,

```bash
python3 prepare_qe_scores_REUSE.py --baseline REUSE
```

The output files will be stored in `scores/qe-as-a-metric`.
