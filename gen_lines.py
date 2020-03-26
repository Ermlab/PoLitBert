
#%%
import sys
import datetime as dt
import os
from pathlib import Path
import re
from tqdm import tqdm 
#from tqdm.notebook import tqdm
import mmap
from text_utils import get_num_lines


#%%

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


                #check if line is valid sentence, if not remove it



                # get block of text to new line which splits ariticles
                text+=line
                i+=1
                if line.strip() == '' or i%1000==0:

                    sentences = sentence_tokenizer.tokenize(text)

                    file_content = ''
                    for sentence in sentences:
                        file_content += sentence.strip()
                        file_content+='\n'
                    output_file.write(file_content)
                    
                    output_file.write('\n')
                    text=''

                
    t1=dt.datetime.now()  
    print(f'Split lines done, takes={t1-t0}')   

