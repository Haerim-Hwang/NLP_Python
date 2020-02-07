##################################################################################
### Step 1: Import modules 
##################################################################################

## Import modules
from __future__ import division # translate the / and /= operators to true division opcodes; use print function without using the parentheses
import string # import "string" to determine the format of the string from its contents
from collections import Counter # create frequency lists
from operator import itemgetter # for sorting
import glob # find and open all the files that match the rules which will be set by the user later
import csv # allow Python to import or export spreadsheets and databases 
from itertools import groupby # group categories and apply a function to the categories (https://www.kaggle.com/crawford/python-groupby-tutorial)
from textblob import TextBlob # process text data
import pandas as pd # analyze data
from lexicalrichness import LexicalRichness # compute lexical richness (https://pypi.org/project/lexicalrichness/#description)
import nltk # import the natural language toolkit
import nltk_tgrep # search matching patterns
from nltk.tree import ParentedTree # create a subclass of tree that automatically maintains parent pointers for single-parented tree 
from nltk.tokenize import sent_tokenize # tokenize text into sentences
import benepar # constituency parser
parser = benepar.Parser("benepar_en") # parse English sentences



##################################################################################
### Step 2: Split one csv file to multiple textfiles by participant code (i.e., row[0])
##################################################################################

for key, rows in groupby(csv.reader(open("data.csv", encoding="utf-8-sig", errors="ignore")), 
                         lambda row: row[0]): # group the data based on the first column (i.e., participant)
    with open("data/%s.txt" % key, "w") as output: # under the "data" folder
        for row in rows: # iterate through rows
            output.write("".join(str(row[1])) + "\t" + (str(row[2])) + "\n") # combine the info under the second column (i.e., utterances) and the info under the third column (i.e., accuracy coding)
            



##################################################################################
### Step 3: Measure (a) Moving-Average Type-Token Ratio and (2) verbal density
##################################################################################

## Open text files under the working directory (e.g., L2 adult group)
filenames = glob.glob("data/*.txt")

output = [] 
output_pos = []

## Set a punctuation list
punct = [",",".","?","'",'"',"!",":",";","(",")","''","``","--", "``", "''"]

for filename in filenames: # iterate through text files in the list
    text = open(filename, "r").read().split("\n") # open the file
    
    utterance_list = [] # create a holder for utterances
    accuracy_coding_list = [] # create a holder for accuracy codings
    
    utt_counter = 0
    
    ## MATTR, Accuracy
    for line in text[:-1]: # iterate through sentences in the text; the last line is blank
        
        line_sep = line.split("\t") # split each line by tab
        utterance_list.append(line_sep[0]) # first part: utterance
        accuracy_coding_list.append(line_sep[1]) # second part: accuracy coding
        utt_counter += 1
        
    ppt_code = filename[5:-4] # get a ppt code
       

    accuracy_int = [int(x) for x in accuracy_coding_list] # get an array of integers; one for each line of input

 
    accuracy = sum(accuracy_int)/len(accuracy_int) # sum of the integers divided by the number of codings
    
    utterances = ' '.join(utterance_list) # make the utterances into text
    
    #output.append([ppt_code, mattr, accuracy]) # append ppt_code, mattr, and accuracy to the "output" list
    
    ## Verbal density
    ## Count the number of verbs
    for z in punct: # replace punctuations with nothing 
        utterances_no_punct = utterances.replace(z, "")
    
    text_nltk = nltk.word_tokenize(utterances_no_punct)# segment a text into basic units--or tokens--such as words and punctuation
    text_pos = nltk.pos_tag(text_nltk) # assign part of speech tags to Brown corpus (note that it must be tokenized first)
       
    verb = []
    verb_counter = 0 
    
    for i, y in enumerate(text_pos):
        #print(i, y)
        if (text_pos[i][1]=='VBD' or text_pos[i][1]=='VBZ') and (text_pos[i+1][1]=='VBG' or text_pos[i+1][1]=='VBN'):
            #print(text_pos[i], text_pos[i+1])
            continue
        elif (text_pos[i][1]=='VB') or (text_pos[i][1]=='VBG') or (text_pos[i][1]=='VBN') or (text_pos[i][1]=='VBP') or (text_pos[i][1]=='VBZ') or (text_pos[i][1]=='VBD'):
            # if the second item in a tuple (pos tag) is one of the verb POS categories,
            #print(text_pos[i])
            #verb.append(text_pos[i][0]) # append the tuple to the verb; to check in case
            verb_counter += 1 # add 1 to the counter    
 
    tunit = len(utterance_list) 
    
    verbal_density = verb_counter/tunit

    lex = LexicalRichness(str(utterance_list)) # instantiate new text object (use use_TextBlob=True argument to use the textblob tokenizer)

	mattr = lex.mattr(window_size = 15)
    
    output.append([ppt_code, mattr, accuracy, verb_counter, tunit, verbal_density]) # put all info into the "output" list
    output = sorted(output, key=itemgetter(0)) # sort the "output" list basaed on the ppt_code

print(output)




##################################################################################
### Step 4: Convert raw scores to z-scores
##################################################################################

## Convert raw_scores to z_scores
df = pd.DataFrame(output, columns = ["participant", "MATTR", "accuracy", "verb", "tunit", "verbal density"]) # set headers for the data frame

## Leave necessary info only
#df = df[["participant", "verbal_density", "mattr", "accuracy"]]

## Compute z-scores
cols = list(df.columns) # get column names
cols.remove('participant') # remove the "participant" column for computing z-scores
cols.remove('verb') # remove the "verb" column for computing z-scores
cols.remove('tunit') # remove the "tunit" column for computing z-scores
df[cols] # check what columns now you have

for col in cols: #iterate through columns
    col_zscore = col + '_zscore' # create column for z-scores
    df[col_zscore] = (df[col] - df[col].mean())/df[col].std(ddof=0) # put zscores

df['proficiency'] = df.iloc[:, df.columns.str.contains('_zscore')].sum(1) # sum z-scores to get a final proficiency score
df # check the data frame



##################################################################################
### Step 5: Save the output in CSV
##################################################################################

df.to_csv("output.csv", sep=',', encoding='utf-8')
