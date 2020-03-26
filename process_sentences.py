#%%
import nltk
import text_utils as tu



#%%
from collections import  namedtuple
import morfeusz2

InterpTuple = namedtuple("InterpTuple", field_names=["word", "lemat", "tags", "info", "style"])

s1= """Operacji tej dokonano w czasie panującego wszechobecnie powojennego chaosu. Zastosowano radykalne lecz efektywne rozwiązanie, do którego zaangażowano DR. Praktycznie jednej nocy z pewnej ilości lokomotyw, wagonów i personelu wydzielono tzw. "Kolonne" (tłum. z niem. jako konwoje), którym powierzono transport sprzętu ze wschodnich Niemiec przez Polskę do granicy radzieckiej.
"""

sentence = 'Adam mieli zboże'

morf = morfeusz2.Morfeusz(separate_numbering=True)
analysis = morf.analyse(sentence)

#%%

for i,a in enumerate(analysis):
    print(i,a)
    inter_tuple = InterpTuple(*a[2])
    lemat = inter_tuple.lemat.split(":")[0]
    


CorpusProcessingTuple = namedtuple("CorpusProcessingTuple", field_names=["file_name", "split_each_line_as_doc", "check_valid_sentence"])

files_to_proces = [
    CorpusProcessingTuple('./data/corpus_wikipedia_2020-02-01.txt', False, False),
    CorpusProcessingTuple('./data/corpus_oscar.txt', True, True ),
    CorpusProcessingTuple('./data/corpus_books.txt', False, False ),
    CorpusProcessingTuple('./data/corpus_subtitles.txt', False, False ),
    CorpusProcessingTuple('./data/corpus_patents.txt', False, True ),
]

#%%

input_file = './data/corpus_raw/corpus_govtech_small.txt'


p = Path(input_file)
output_path = f"{p.with_suffix('')}_lines.txt"


sentence_tokenizer = tu.create_nltk_sentence_tokenizer()
total_lines = tu.get_num_lines(input_file)

with open(output_path, 'w+') as output_file:
        with open(input_file) as f:
            i=0
            for line in tqdm(f,total=total_lines):

                # get block of text to new line which splits ariticles
                text+=line
                i+=1
                if line.strip() == '' | i%100==0:

                    sentences = sentence_tokenizer.tokenize(text)

                    file_content = ''
                    for sentence in sentences:
                        file_content += sentence.strip()
                        file_content+='\n'
                    output_file.write(file_content)
                    
                    output_file.write('\n')
                    text=''
                    print(f'{i}/{total_lines}={i/total_lines}')

         