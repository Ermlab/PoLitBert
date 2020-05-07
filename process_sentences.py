#%%
import text_utils as tu

from pathlib import Path
from tqdm import tqdm

from collections import namedtuple

#%%
CorpusProcessingTuple = namedtuple(
    "CorpusProcessingTuple",
    field_names=["file_path", "split_each_line_as_doc", "check_valid_sentence"],
)

# files_to_proces = [
#     CorpusProcessingTuple('./data/corpus_wikipedia_2020-02-01.txt', False, False),
#     CorpusProcessingTuple('./data/corpus_oscar.txt', True, True ),
#     CorpusProcessingTuple('./data/corpus_books.txt', False, False ),
#     CorpusProcessingTuple('./data/corpus_subtitles.txt', False, False ),
#     CorpusProcessingTuple('./data/corpus_patents.txt', False, True ),
# ]


# corpus_tuple =CorpusProcessingTuple('./data/corpus_raw/corpus_govtech_small.txt', False, True)
corpus_tuple = CorpusProcessingTuple(
    "./data/corpus_raw/corpus_oscar_100k.txt", True, True
)


input_file = corpus_tuple.file_path
p = Path(input_file)
output_file = f"{p.with_suffix('')}_lines_sentence_pl.txt"

print(f"in file={input_file}\nout file={output_file}")

stats, vl, pl = tu.corpus_process_sentence(
    input_file,
    output_file,
    split_each_line_as_doc=corpus_tuple.split_each_line_as_doc,
    check_valid_sentence=corpus_tuple.check_valid_sentence,
    max_sentence_length=700,
)
#%%
from pprint import pprint

print(f"\nStatystyki=")
pprint(stats)

#%%

