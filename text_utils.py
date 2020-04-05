"""Text utils

"""
import sys
import datetime as dt
import os
import mmap
from pathlib import Path
from tqdm import tqdm

from collections import namedtuple

import morfeusz2
import nltk

from langdetect import detect_langs

from polyglot.detect import Detector


def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines


import logging


logging.basicConfig(level=logging.ERROR)
# disable logging from polyglot
#Detector is not able to detect the language reliably.
logger_poly = logging.getLogger('polyglot.detect.base:Detector')

logger_poly.setLevel(level=logging.CRITICAL)
logger_poly.propagate = False
logger_poly.disabled = True


def check_polish_sentence(sentence):
    """Returns true if sentence is written in polish
    Uses langdetect library and polyglot.
    
    """


    # prevent error "input contains invalid UTF-8 around byte"
    text_to_detect = ''.join(x for x in sentence if x.isprintable())
    try:
        langs = detect_langs(text_to_detect)
    except Exception as e:
        # print(f"<{sentence}> - exp={e}")
        # if we cant detect the language, return True and try to deal with it later in the pipeline
        return True

    # if contains 'pl' with probability grateher then 0.4
    langdet_pl =  any([(l.lang == "pl" and l.prob > 0.4) for l in langs])

    detector = Detector(text_to_detect, quiet=True)
    poly_pl = detector.language.code =='pl' and detector.language.confidence>40

    return langdet_pl or poly_pl



def create_nltk_sentence_tokenizer():

    extra_abbreviations = [
        "ps",
        "inc",
        "Corp",
        "Ltd",
        "Co",
        "pkt",
        "Dz.Ap",
        "Jr",
        "jr",
        "sp.k",
        "sp",
        # "Sp",
        "poj",
        "pseud",
        "krypt",
        "sygn",
        "Dz.U",
        "ws",
        "itd",
        "np",
        "sanskryt",
        "nr",
        "gł",
        "Takht",
        "tzw",
        "t.zw",
        "ewan",
        "tyt",
        "fig",
        "oryg",
        "t.j",
        "vs",
        "l.mn",
        "l.poj",
        "ul",
        "al",
        "Al",
        "el",
        "tel",
        "wew",
        "bud",
        "pok",
        "wł",
    ]

    position_abbrev = [
        "Ks",
        "Abp",
        "abp",
        "bp",
        "dr",
        "kard",
        "mgr",
        "prof",
        "zwycz",
        "hab",
        "arch",
        "arch.kraj",
        "B.Sc",
        "Ph.D",
        "lek",
        "med",
        "n.med",
        "bł",
        "św",
        "hr",
        "dziek",
    ]

    roman_abbrev = (
        []
    )  # ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII', 'XII','XIV','XV','XVI', 'XVII', 'XVIII','XIX', 'XX', 'XXI' ]

    quantity_abbrev = [
        "mln",
        "obr./min",
        "km/godz",
        "godz",
        "egz",
        "ha",
        "j.m",
        "cal",
        "obj",
        "alk",
        "wag",
    ]  # not added: tys.

    actions_abbrev = [
        "tłum",
        "tlum",
        "zob",
        "wym",
        "pot",
        "ww",
        "ogł",
        "wyd",
        "min",
        "m.i",
        "m.in",
        "in",
        "im",
        "muz",
        "tj",
        "dot",
        "wsp",
        "właść",
        "właśc",
        "przedr",
        "czyt",
        "proj",
        "dosł",
        "hist",
        "daw",
        "zwł",
        "zaw",
        "późn",
        "spr",
    ]

    place_abbrev = ["Śl", "płd", "geogr"]

    lang_abbrev = [
        "jęz",
        "fr",
        "franc",
        "ukr",
        "ang",
        "gr",
        "hebr",
        "czes",
        "pol",
        "niem",
        "arab",
        "egip",
        "hiszp",
        "jap",
        "chin",
        "kor",
        "tyb",
        "wiet",
        "sum",
        "chor",
        "słow",
        "węg",
        "ros",
        "boś",
    ]

    military_abbrev = [
        "kpt",
        "kpr",
        "obs",
        "pil",
        "mjr",
        "płk",
        "dypl",
        "pp",
        "gw",
        "dyw",
        "bryg",
        "ppłk",
        "mar",
        "marsz",
        "rez",
        "ppor",
        "DPanc",
        "BPanc",
        "DKaw",
        "p.uł",
    ]

    extra_abbreviations = (
        extra_abbreviations
        + position_abbrev
        + roman_abbrev
        + quantity_abbrev
        + place_abbrev
        + actions_abbrev
        + place_abbrev
        + lang_abbrev
        + military_abbrev
    )

    # create tokenizer with update abrev
    sentence_tokenizer = nltk.data.load("tokenizers/punkt/polish.pickle")
    # update abbrev
    sentence_tokenizer._params.abbrev_types.update(extra_abbreviations)

    return sentence_tokenizer


def corpus_process_sentence(
    corpus_input_file,
    courpus_output_file,
    split_each_line_as_doc,
    check_valid_sentence=False,
    max_sentence_length=5000,
):
    """
    Read corpus_input_file and save each sentence to new line of output file. Do some checks:
    if sentence is not to long, to short, if is written in polish, and if is valid polish sentence
    conaining verb

        :param corpus_input_file: path to corpus txt file
        :param output_file: path to output txt file where each sentence is in seperate line
        :param split_each_line_as_doc: determine if treat each line as separate document, if so add additional blank line after the end of the string, if set to false each document has already new blank line at the end 
        :param check_valid_sentence: keep only valid sentence, those with verb, omit non valid sentences
        :param max_sentence_length: max sentence length in characters, omit sentece longer then 1500, high probability of non valid sentence
    """

    total_lines = get_num_lines(corpus_input_file)
    sentence_tokenizer = create_nltk_sentence_tokenizer()
    morf_sent = MorfAnalyzer()
    # statistics
    invalid_length_sentences = 0
    non_valid_sentences = 0
    all_sentences = 0
    non_polish = 0

    with open(courpus_output_file, "w+") as output_file:
        with open(corpus_input_file) as f:
            i = 0
            text = ""
            for line in tqdm(f, total=total_lines):

                # get block of text to new line which splits articles
                text += line

                i += 1
                if split_each_line_as_doc or line.strip() == "" or i % 100 == 0:
                    # if split_each_line_as_doc is set then add new line after each line, if not then read file up to empty line (or max 100 lines)
                    sentences = sentence_tokenizer.tokenize(text)

                    file_content = ""
                    for sentence in sentences:

                        sentence = sentence.strip()
                        sentence_length = len(sentence)

                        all_sentences += 1

                        if (
                            sentence_length < 4
                            or sentence_length > max_sentence_length
                        ):
                            # omit to long and too short sentences
                            invalid_length_sentences += 1
                            continue

                        if not check_polish_sentence(sentence):
                            non_polish += 1
                            continue

                        if (
                            check_valid_sentence
                            and sentence_length > 80
                            and not morf_sent.sentence_valid(sentence)
                        ):

                            # omit sentence if is not valid, we do not check short sentences
                            non_valid_sentences += 1

                            continue

                        file_content += sentence
                        file_content += "\n"

                    if file_content != "":

                        output_file.write(file_content)
                        output_file.write("\n")

                    text = ""

    stats = {
        "lines": i,
        "all_sentences": all_sentences,
        "non_valid_sentences": non_valid_sentences,
        "invalid_length_sentences": invalid_length_sentences,
        "non_polish": non_polish,
    }

    return stats


InterpTuple = namedtuple(
    "InterpTuple", field_names=["word", "lemat", "tags", "info", "style"]
)


class MorfAnalyzer(object):
    def __init__(self):
        self._morfeusz = morfeusz2.Morfeusz(separate_numbering=True)

        self._verb_pattern = set(["fin", "praet", "inf", "pred", "bedzie"])  # 'ger'

    def analyse(self, sentence):
        """Analyse the sentence and return morfeusz2 morf tags
        """
        analysis = self._morfeusz.analyse(sentence)
        return analysis

    def sentence_valid(self, txt):
        """Check if the passed txt is valid sentence, should contain min. one verb in proper form"""

        analysis = self.analyse(txt)

        # evaluation is done lazy
        return any(
            InterpTuple(*a[2]).tags.split(":")[0] in self._verb_pattern
            for a in analysis
        )
