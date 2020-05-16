#%%
import text_utils as tu

from pathlib import Path
from tqdm import tqdm
import datetime as dt
from collections import namedtuple

import argparse


#%%


parser = argparse.ArgumentParser(
    description='Process raw corpus txt file, split to lines, check is sentence is valid, remove to  short and to long sentences and save to file with suffix "_lines"'
)
parser.add_argument(
    "corpus_file", type=str, help="corpus txt raw input file"
)


parser.add_argument(
    "-sp",
    "--split_each_line_as_doc",
    action="store_true",
    help="If true each line from corpus file will be treated as document, new line will be added after last sentence from this line",
    default=False,
)

parser.add_argument(
    "-vs",
    "--check_valid_sentence",
    action="store_true",
    default=False,
    help="check if extracted sentence is valid polish sentence, if not do not save it in output file",
)



parser.add_argument(
    "-ls",
    "--check_lang_sentence",
    action="store_true",
    default=False,
    help="check if extracted sentence is in polish, remove sentences in other lang, do not save it in output file",
)

parser.add_argument(
    "-ml",
    "--max_sentence_length",
    type=int,
    default=700,
    help="remove longer(in chars) sentences",
)


args = parser.parse_args()


#%%

corpus_oscar_raw = args.corpus_file



p = Path(corpus_oscar_raw)
corpus_oscar_lines = f"{p.with_suffix('')}_lines.txt"

print(f"Start preparing corpus")
print(f"in file={corpus_oscar_raw}\nout file={corpus_oscar_lines}")
start = dt.datetime.now()
print(f"Start time: {start}")


stats, vl, pl = tu.corpus_process_sentence(
    corpus_oscar_raw,
    corpus_oscar_lines,
    split_each_line_as_doc=args.split_each_line_as_doc,
    check_valid_sentence=args.check_valid_sentence,
    check_lang_sentence=args.check_lang_sentence,
    max_sentence_length=args.max_sentence_length,
)

end = dt.datetime.now()
print(f"Finish. End time: {end} Start time: {start} took={end-start}")

from pprint import pprint

print(f"Cleaning stats")
pprint(stats)

#%%
