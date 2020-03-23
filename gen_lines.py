
#%%
import csv
import sys
import datetime as dt
import os
from pathlib import Path
import re
from tqdm import tqdm 
#from tqdm.notebook import tqdm
import mmap

#%% utils functions
def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines


#%%
import nltk

#nltk.download()
#nltk.download('punkt')

extra_abbreviations = ['ps',  'inc', 'Corp', 'Ltd', 'Co', 'pkt', 'Dz.Ap', 'Jr', 'jr', 'sp', 'Sp', 'poj',  'pseud', 'krypt', 'sygn', 'Dz.U', 'ws', 'itd', 'np', 'sanskryt', 'nr', 'gł', 'Takht', 'tzw', 't.zw', 'ewan', 'tyt', 'oryg', 't.j', 'vs', 'l.mn', 'l.poj' ]

position_abbrev = ['Ks', 'Abp', 'abp','bp','dr', 'kard', 'mgr', 'prof', 'zwycz', 'hab', 'arch', 'arch.kraj', 'B.Sc', 'Ph.D', 'lek', 'med', 'n.med', 'bł', 'św', 'hr', 'dziek' ]

roman_abbrev= [] #['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII', 'XII','XIV','XV','XVI', 'XVII', 'XVIII','XIX', 'XX', 'XXI' ]

quantity_abbrev = [ 'mln', 'obr./min','km/godz', 'godz', 'egz', 'ha', 'j.m', 'cal', 'obj', 'alk', 'wag' ] # not added: tys.

actions_abbrev = ['tłum','tlum','zob','wym', 'pot', 'ww', 'ogł', 'wyd', 'min', 'm.i', 'm.in', 'in', 'im','muz','tj', 'dot', 'wsp', 'właść', 'właśc', 'przedr', 'czyt', 'proj', 'dosł', 'hist', 'daw', 'zwł', 'zaw' ]

place_abbrev = ['Śl', 'płd', 'geogr']

lang_abbrev = ['jęz','fr','franc', 'ukr', 'ang', 'gr', 'hebr', 'czes', 'pol', 'niem', 'arab', 'egip', 'hiszp', 'jap', 'chin', 'kor', 'tyb', 'wiet', 'sum', 'chor', 'słow', 'węg', 'ros', 'boś']

military_abbrev = ['kpt', 'kpr', 'obs', 'pil', 'mjr','płk', 'dypl', 'pp', 'gw', 'dyw', 'bryg', 'ppłk', 'mar', 'marsz', 'rez', 'ppor', 'DPanc', 'BPanc', 'DKaw', 'p.uł']

extra_abbreviations= extra_abbreviations + position_abbrev + roman_abbrev + quantity_abbrev + place_abbrev + actions_abbrev + place_abbrev + lang_abbrev+military_abbrev

sentence_tokenizer = nltk.data.load('tokenizers/punkt/polish.pickle')

abbrev_set_curr = sentence_tokenizer._params.abbrev_types.copy()
#update abbrev
sentence_tokenizer._params.abbrev_types.update(extra_abbreviations)
abbrev_set_upd = sentence_tokenizer._params.abbrev_types.copy()

# print(f'\ncurrent abbrev={abbrev_set_curr}')
sentence_tokenizer._params.abbrev_types.update(extra_abbreviations)
# print(f'\nupdated abbrev={abbrev_set_upd}')

# print(f'symmetric diff= {abbrev_set_upd.symmetric_difference(abbrev_set_curr)}')

# print(f'Updatad diff= {abbrev_set_upd.difference(abbrev_set_curr)}')
# print(f'Curr diff= {abbrev_set_curr.difference(abbrev_set_upd)}')



files_to_proces = [
    './data/corpus_wikipedia_2020-02-01.txt',
    './data/corpus_books_2020_02_24.txt'
]
files_to_proces = [
    './data/corpus_wikipedia_2020-02-01.txt',
]
files_to_proces = [
    './data/corpus_books_2020_02_24_fix.txt'
]

from nltk.tokenize import sent_tokenize

for input_file in files_to_proces:
    print(input_file)
    p = Path(input_file)
    output_path = f"{p.with_suffix('')}_lines.txt"

    print(f"in file={input_file}\nout file={output_path}")

    t0=dt.datetime.now()

    total_lines = get_num_lines(input_file)
    
    text=''
    with open(output_path, 'w+') as output_file:
        with open(input_file) as f:
            i=0
            for line in tqdm(f,total=total_lines):

                # get block of text to new line which splits ariticles
                text+=line
                i+=1
                if line.strip() == '' or i%10000==0:

                    sentences = sentence_tokenizer.tokenize(text)

                    file_content = ''
                    for sentence in sentences:
                        file_content += sentence.strip()
                        file_content+='\n'
                    output_file.write(file_content)
                    
                    output_file.write('\n')
                    text=''
                    #print(f'{i}/{total_lines}={i/total_lines}')

                ## old way
                # if line.strip() == '':
                #     output_file.write('\n')
                #     continue;

                # sentences = sentence_tokenizer.tokenize(line)
                
                # text = ''
                # for sentence in sentences:
                #     text += sentence.strip()
                #     text+='\n'
                # output_file.write(text)
                
    t1=dt.datetime.now()  
    print(f'Split lines done, takes={t1-t0}')   

