

#%%
import sys
import datetime as dt
import os
from pathlib import Path
import pandas as pd

from tqdm import tqdm
#from tqdm.notebook import tqdm
import mmap


# utils functions


def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines

#%%
input_file = './data/wiki_model/corpus_wikipedia_2020-02-01.txt.bpe'
input_file='./data/wiki_model/corpus_wikipedia_2020-02-01_train.txt.bpe'

#%%
number_of_tokens = []

with open(input_file) as f:
    for line in tqdm(f, total=get_num_lines(input_file)):
        number_of_tokens.append(len(line.strip().split()))


#%%

df = pd.DataFrame(number_of_tokens)


bins = [0,100,200,300,400,512,10000]

print(df[0].value_counts(bins=bins))

df.hist(bins=bins)


print(df[df[0]>511].count())



# %%
