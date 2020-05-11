##################################################################################
### Step 1: Import modules 
##################################################################################

## Import modules
from collections import Counter # count the number of matches
import csv # save output in CSV 
from operator import itemgetter # sort
import glob # find and open all the files that match the rules which will be set by the user later

import nltk # natural language toolkit
import nltk_tgrep # search matching patterns
from nltk.tree import ParentedTree # create a subclass of tree that automatically maintains parent pointers for single-parented tree 
from nltk.tokenize import sent_tokenize # tokenize text into sentences

import benepar # constituency parser
parser = benepar.Parser("benepar_en") # parse English sentences

import spacy # part-of-speech tagger and dependency parser
nlp = spacy.load('en_core_web_sm') # parse English sentences 





##################################################################################
### Step 2: Split one csv file to multiple textfiles by participant code (i.e., row[0])
##################################################################################

from itertools import groupby # group categories and apply a function to the categories (https://www.kaggle.com/crawford/python-groupby-tutorial)

for key, rows in groupby(csv.reader(open("Kor-teacher_data/FLKC.csv", encoding="utf-8-sig", errors="ignore")), 
                         lambda row: row[0]): # for the csv file; group the data based on the first column (i.e., teacher code)

    with open("Kor-teacher_data/%s.txt" % key, "w") as output: # under the data folder
        for row in rows: # iterate through rows            
            output.write("".join(str(row[1])) + "\n") # combine the info under the second column (i.e., utterances)



##################################################################################
### Step 3: Identify VPE candididates
##################################################################################

## Sub_step 1: Set the lemma triggers and POS tag triggers
trigger = "are is was were be can could do does did has have may might shall should to will would not n't".split(" ") # filter 1: set words that potentially triggers VP-ellipsis
pos_trigger = ["VERB", "AUX", "XCOMP", "ADP", "NEG", "NOUN", "ADJ", "PROPN", "NUM", "DET", "PRON"] # filter 2: set categories that cannot come after the triggers to be VP-ellipsis 
tagq_trigger = ["PRON"] # set the trigger of a tag question
q_trigger = ["who", "what", "when", "where", "how", "why"]


## Sub_step 2: Extract VP-ellipsis candidates
vpe_candidates = [] # create a holder

all_files = sorted(glob.glob("practice_data/*.txt")) # use glob to get all filenames in folder


for file in all_files: # iterate through files
    text = open(file, "r", encoding="utf-8-sig").read() # open files 

    total_counter = 0 # set the total frequency of VPE candidates as “0”
    text_tokenized = sent_tokenize(text) # text to sentence tokens   
    #print(text_tokenized)
    
    for sent in text_tokenized: # iterate through sentences
        doc = nlp(sent) # process text with spacy (part-of-speech tagger and dependency parser)

        for n in range(len(doc)): # iterate through tokens parsed
            token = doc[n-1] # previous token
            token_next = doc[n] # the current token
            
            if token.lemma_ in trigger: # if the previous token's lemma is in the "trigger" list 
                
                
                if len(doc) >= 3: # if a sentence length including a punctuation mark is more than 2 (e.g., you did.)
                    
                    ## VPE in tag questions
                    if doc[-3].lemma_ in trigger and doc[-2].pos_ in tagq_trigger: # if the second last word's lemma is in the list "trigger" and the last word's category is in the list "tagq_trigger"
                        total_counter += 1 # add the number of matches to the "total_counter"
                        VPE_type ="VPE_tag_candidates" # mark it as "VPE_tag_candidates"
                        vpe_trigger_list = [file, VPE_type, total_counter, token, sent] # create the "vpe_trigger_list" 
                        vpe_candidates.append(vpe_trigger_list) # append the above info to the list "vpe_candidates"                
                        print(vpe_trigger_list) 

                                   
                    ## VPE in declaratives
                    elif token_next.pos_ not in pos_trigger: # if the category of the current token is not in "pos_trigger"
                        total_counter += 1 # add the number of matches to the "total_counter"
                        VPE_type ="VPE_decl_candidates"
                        vpe_trigger_list = [file, VPE_type, total_counter, token, sent] # create the "vpe_trigger_list" 
                        vpe_candidates.append(vpe_trigger_list) # append the above info to the list "vpe_candidates"                
                        print(vpe_trigger_list)
                    
                    else:
                        continue
                    
                    ## create/open csv file to save the data
                    import os.path
                    vpe_candidates_exists = os.path.isfile("xxx.csv") # set the file path
                    with open ("xxx.csv", 'a') as csvfile: # create the file
                        headers = ["file_name", "total_counter", "counter", "token", "VPE_type", "sent"] # set the headers
                        writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n',fieldnames=headers)
                        if not vpe_candidates_exists: # if the file doesn't exist yet
                            writer.writeheader()  # write headers
                        writer.writerow({"file_name": file, 'total_counter': total_counter, 'counter': "1", 'token': token, "VPE_type": VPE_type, "sent": sent}) # write a CSV file including the file name, the total number of VP-ellipsis candidates, “1” as the number of VP-ellipsis candidate identified, type of a VP-ellipsis candidate (e.g., VP-ellipsis in a tag question or VP-ellipsis in other patterns), and the sentence identified as a VP-ellipsis candidate
                
                else:
                    continue
                    
            else:
                continue
                




##################################################################################
### Step 4: Identify Gapping candididates
##################################################################################

## Sub_step 1: Set the pattern which potentially has Gapping in it
gap = 'NP|NN|NNP|PP $ NN|NNP|NP|PP|ADVP !$ VP|VBD|VBG|VBN|VBP|VBZ|VB'# potential gapping sentences: If NP is a sister of other constituents except for VPs


## Sub_step 2: Extract Gapping candidates
all_files = sorted(glob.glob("practice_data/*.txt")) # use glob to get all filenames in folder

for file in all_files: # iterate through files
    text = open(file, "r", encoding="utf-8-sig").read() # open each file 
    
    file_name = file
    total_counter = 0 # set the total frequency of gapping candidates as "0"
    
    text_tokenized = sent_tokenize(text) # text to sentence tokens 
    
    for sent in text_tokenized: # iterate through sentences
        sent = sent.replace(",", " ") # remove commas
        sent_tree = parser.parse(sent) # parse sentences
        sent_tree = ParentedTree.fromstring(str(sent_tree)) # convert trees to the appropriate format for natural language processing
        match = nltk_tgrep.tgrep_nodes(sent_tree, gap) # find matches to the pattern named "gap" above
        
        #con_sub_match = nltk_tgrep.tgrep_nodes(sent_tree, con_sub) # find matches to the pattern named "con_sub"
        #if len(con_sub_match) != 0: # if there is one match or more to the pattern "con_sub"
        
        if len(match) != 0:  # if there is one match or more to the pattern "gap"
            total_counter += len(match) # add the number of matches to the "total_counter"
            counter = len(match) # get the number of matches
        else:
            continue # if not, continue
                
        print(total_counter, counter, sent) # print the output
            
        ## create/open csv file to save the freq data
        import os.path
        gapping_candidates_exists = os.path.isfile("gapping_candidates_textbook.csv") # set the file path
        with open ("gapping_candidates_textbook.csv", 'a') as csvfile: # create a file
            headers = ["file_name", "total_counter", "counter", "sent"] # set headers
            writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n',fieldnames=headers) 
            if not gapping_candidates_exists:
                writer.writeheader()  # if a file doesn't exist yet, write a header
            writer.writerow({"file_name": file_name, 'total_counter': total_counter, 'counter': counter, "sent": sent})# write a CSV file including the file name, the total number of Gapping candidates, the number of Gapping candidates in the sentence under analysis, and the sentence identified as a Gapping candidate


