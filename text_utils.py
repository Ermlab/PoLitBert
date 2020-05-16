"""Text utils

"""
import json
import requests
from enum import Enum
import stanza
from abc import ABC, abstractmethod
import logging
import sys
import datetime as dt
import os
import mmap
from pathlib import Path
from tqdm import tqdm

from collections import namedtuple, Counter
import itertools


#import morfeusz2
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


logging.basicConfig(level=logging.ERROR)
# disable logging from polyglot
# Detector is not able to detect the language reliably.
logger_poly = logging.getLogger("polyglot.detect.base:Detector")

logger_poly.setLevel(level=logging.CRITICAL)
logger_poly.propagate = False
logger_poly.disabled = True


def check_polish_sentence(sentence):
    """Returns true if sentence is written in polish
    Uses langdetect library and polyglot.

    """

    # prevent error "input contains invalid UTF-8 around byte"
    text_to_detect = "".join(x for x in sentence if x.isprintable())
    try:
        langs = detect_langs(text_to_detect)
    except Exception as e:
        # print(f"<{sentence}> - exp={e}")
        # if we cant detect the language, return True and try to deal with it later in the pipeline
        return True

    # if contains 'pl' with probability grateher then 0.4
    langdet_pl = any([(l.lang == "pl" and l.prob > 0.4) for l in langs])

    detector = Detector(text_to_detect, quiet=True)
    poly_pl = detector.language.code == "pl" and detector.language.confidence > 40

    return langdet_pl or poly_pl


def create_nltk_sentence_tokenizer():
    """

    find in vim with pattern: /\<\w\{2,3\}\>\.\n
    """

    extra_abbreviations = [
        "ps",
        "inc",
        "corp",
        "ltd",
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
        "ws",
        "itd",
        "np",
        "sanskryt",
        "nr",
        "gł",
        "Takht",
        "tzw",
        "tzn",
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
        "wew",  # wewnętrzny
        "bud",
        "pok",
        "wł",
        "sam",  # samochód
        "sa",  # spółka sa.
        "wit",  # witaminy
        "mat",  # materiały
        "kat",  # kategorii
        "wg",  # według
        "btw",  #
        "itp",  #
        "wz",  # w związku
        "gosp",  #
        "dział",  #
        "hurt",  #
        "mech",  #
        "wyj",  # wyj
        "pt",  # pod tytułem
        "zew",  # zewnętrzny
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
        "obr",  # obroty
        "wk",
        "mm",
        "MB",  # mega bajty
        "Mb",  # mega bity
        "jedn",  # jednostkowe
        "op",
        "szt",  # sztuk
    ]  # not added: tys.

    actions_abbrev = [
        "tłum",
        "tlum",
        "zob",
        "wym",
        "w/wym",
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
        "jw",
        "odp",  # odpowiedź
        "symb",  # symbol
        "klaw",  # klawiaturowe
    ]

    place_abbrev = [
        "śl",
        "płd",
        "geogr",
        "zs",
        "pom",  # pomorskie
        "kuj-pom",  # kujawsko pomorskie
    ]

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
        "szw",
    ]

    administration = [
        "dz.urz",  # dziennik urzędowy
        "póź.zm",
        "rej",  # rejestr, rejestracyjny dowód
        "sygn",  # sygnatura
        "Dz.U",  # dziennik ustaw
        "woj",  # województow
        "ozn",  #
        "ust",  # ustawa
        "ref",  # ref
        "dz",
        "akt",  # akta
    ]

    time = [
        "tyg",  # tygodniu
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
        "bryg",  # brygady
        "ppłk",
        "mar",
        "marsz",
        "rez",
        "ppor",
        "DPanc",
        "BPanc",
        "DKaw",
        "p.uł",
        "sierż",
        "post",
        "asp",
        "szt",  # sztabowy
        "podinsp",
        "kom",  # komendant, tel. komórka
        "nadkom"
    ]

    extra_abbreviations = (
        extra_abbreviations
        + position_abbrev
        + quantity_abbrev
        + place_abbrev
        + actions_abbrev
        + lang_abbrev
        + administration
        + time
        + military_abbrev
    )

    # create tokenizer with update abrev
    sentence_tokenizer = nltk.data.load("tokenizers/punkt/polish.pickle")
    # update abbrev
    sentence_tokenizer._params.abbrev_types.update(extra_abbreviations)

    return sentence_tokenizer


# def sent_tokenizer(sentence):


#     return


def corpus_process_sentence(
    corpus_input_file,
    courpus_output_file,
    split_each_line_as_doc,
    check_valid_sentence=False,
    check_lang_sentence=False,
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

    # morf_sent = MorfeuszAnalyzer()
    # morf_sent = StanzaAnalyzer()
    morf_sent = KRNNTAnalyzer()

    # statistics
    invalid_length_sentences = 0
    non_valid_sentences = 0
    all_sentences = 0
    non_polish = 0

    non_valid_sentences_list = []
    non_polish_list = []

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

                        if sentence_length < 4 or sentence_length > max_sentence_length:
                            # omit to long and too short sentences
                            invalid_length_sentences += 1
                            continue

                        if (check_lang_sentence and sentence_length > 40 and not check_polish_sentence(sentence)):
                            non_polish += 1
                            non_polish_list.append(sentence)
                            continue

                        if (
                            check_valid_sentence
                            and sentence_length > 60
                            and not morf_sent.sentence_valid(sentence)
                        ):

                            # omit sentence if is not valid, we do not check short sentences
                            non_valid_sentences += 1
                            non_valid_sentences_list.append(sentence)

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

    return stats, non_valid_sentences_list, non_polish_list


class MorfAnalyzer(ABC):
    def __init__(self):
        super(MorfAnalyzer, self).__init__()

    @abstractmethod
    def analyse(self, sentence):
        """Analyse the sentence and return  morf tags"""

    @abstractmethod
    def sentence_valid(self, sentence):
        """Check if the passed txt is valid sentence, should contain min. one verb in proper form"""
        pass


InterpTuple = namedtuple(
    "InterpTuple", field_names=["word", "lemat", "tags", "info", "style"]
)


class MorfeuszAnalyzer(MorfAnalyzer):
    def __init__(self):
        super(MorfeuszAnalyzer, self).__init__()

        self._morfeusz = morfeusz2.Morfeusz(separate_numbering=True)
        self._verb_pattern = set(
            ["fin", "praet", "inf", "pred", "impt", "imps", "bedzie"]
        )  # 'ger', ppas

    def analyse(self, sentence):
        """Analyse the sentence and return morfeusz2 morf tags
        """
        analysis = self._morfeusz.analyse(sentence)
        return analysis

    def sentence_valid(self, sentence):
        """Check if the passed txt is valid sentence, should contain min. one verb in proper form"""

        analysis = self.analyse(sentence)

        # evaluation is done lazy
        return any(
            InterpTuple(*a[2]).tags.split(":")[0] in self._verb_pattern
            for a in analysis
        )


class StanzaAnalyzer(MorfAnalyzer):
    def __init__(self):
        super(StanzaAnalyzer, self).__init__()

        stanza.download("pl")
        self._nlp_pipeline = stanza.Pipeline(
            "pl", processors="tokenize,pos,lemma", verbose=True, use_gpu=True
        )  # initialize neural pipeline

        self._conv_stanza_pos = lambda x: [w.pos for w in x.words]

    def analyse(self, sentence):
        """Analyse the sentence and return stanza pos tags
        """
        return self._nlp_pipeline(sentence)

    def sentence_valid(self, sentence):
        """Check if the passed txt is valid sentence, should contain min. one verb in proper form"""

        doc = self.analyse(sentence)

        flatten = itertools.chain.from_iterable

        # get flatten list of tokens from all sentences tokenized by stanza
        # our sentence tokenization is different from stanza, very often our tokenized
        # sentence is treated as 2 or 3 sentences by stanza
        # map sentence word to pos tags and flatten all list
        stanza_pos = list(flatten(map(self._conv_stanza_pos, doc.sentences)))
        stats_stanza_pos = Counter(stanza_pos)

        # prosta heurystyka na bazie obserwacji
        # musi być min. 1 VERB
        # 1 VERB - max_NOUN 7-10 NOUN
        # 2 VERB - max_noun+2
        # 3 verb - max_noun+4 itp

        verbs = stats_stanza_pos["VERB"]
        nouns = stats_stanza_pos["NOUN"] + \
            stats_stanza_pos["PROPN"] + stats_stanza_pos["DET"]
        aux = stats_stanza_pos["AUX"]

        # aux can be treated in some sentences as sentence builder
        verbs = verbs + aux

        # max number of nouns coresponding to first verb
        max_noun = 12
        # additional nouns to additional verbs
        nouns_per_verb = 2

        if verbs < 1:
            # if sentence does not contain any verb then is not valid
            return False
        elif nouns <= max_noun + (verbs - 1) * nouns_per_verb:
            return True
        else:
            return False


class KRNNTAnalyzer(MorfAnalyzer):
    def __init__(self):
        super(KRNNTAnalyzer, self).__init__()

        # docker run -p 9003:9003 -it djstrong/krnnt:1.0.0
        self._url = "http://localhost:9003/?output_format=jsonl"

        self._conv_main_nkjp = lambda x: x[2].split(":")[0]
        self._conv_main_ud = lambda x: get_main_ud_pos(x[2])

    def analyse(self, sentence):
        """Analyse the sentence and return nkjp pos tags
        """

        x = requests.post(self._url, data=sentence.encode("utf-8"))
        resp = x.json()
        return resp

    def sentence_valid(self, sentence):
        """Check if the passed txt is valid sentence, should contain min. one verb in proper form"""

        resp = self.analyse(sentence)

        krnnt_pos = list(map(self._conv_main_ud, resp[0]))

        stats_krnnt_ud = Counter(krnnt_pos)

        # prosta heurystyka na bazie obserwacji
        # musi być min. 1 VERB
        # 1 VERB - max_NOUN 7-10 NOUN
        # 2 VERB - max_noun+2
        # 3 verb - max_noun+4 itp

        verbs = stats_krnnt_ud["VERB"]

        # nouns + unknown words + "uch, ech, psst itp"
        nouns = stats_krnnt_ud["NOUN"] + \
            stats_krnnt_ud["X"] + stats_krnnt_ud["INTJ"]
        aux = stats_krnnt_ud["AUX"]

        # aux can be treated in some sentences as sentence builder
        verbs = verbs + aux

        # max number of nouns coresponding to first verb
        max_noun = 12
        # additional nouns to additional verbs
        nouns_per_verb = 2

        if verbs < 1:
            # if sentence does not contain any verb then is not valid
            return False
        elif nouns <= max_noun + (verbs - 1) * nouns_per_verb:
            return True
        else:
            return False


# mapping from nkjp to ud tag set
# https://gitlab.com/piotr.pezik/apt_pl/-/blob/master/translation.py

# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

D_FIELD = Enum(
    "D_FIELD", "flexemes cats special_lemmas special_words default POS FEATURES"
)


nkjp_to_ud_dict = {
    D_FIELD.flexemes.name: {
        "ger": {
            D_FIELD.default.name: {
                D_FIELD.POS.name: "NOUN",
                D_FIELD.FEATURES.name: [("VerbForm", "Ger")],
            }
        },
        "subst": {
            D_FIELD.special_lemmas.name: [
                (
                    ["kto", "co"],  # Kto i co są w subst-ach
                    {
                        D_FIELD.POS.name: "PRON",
                        D_FIELD.FEATURES.name: [("PronType", "Int,Rel")],
                    },
                ),
                (
                    ["coś", "ktoś", "cokolwiek", "ktokolwiek"],
                    {
                        D_FIELD.POS.name: "PRON",
                        D_FIELD.FEATURES.name: [("PronType", "Ind")],
                    },
                ),
                (
                    ["nikt", "nic"],  # nikt i nic to subst
                    {
                        D_FIELD.POS.name: "PRON",
                        D_FIELD.FEATURES.name: [("PronType", "Neg")],
                    },
                ),
                (
                    ["wszystko", "wszyscy"],
                    {
                        D_FIELD.POS.name: "PRON",
                        D_FIELD.FEATURES.name: [("PronType", "Tot")],
                    },
                ),
                (
                    ["to"],
                    {
                        D_FIELD.POS.name: "PRON",
                        D_FIELD.FEATURES.name: [("PronType", "Dem")],
                    },
                ),
            ],
            D_FIELD.default.name: {
                D_FIELD.POS.name: "NOUN",
                # D_FIELD.FEATURES.name:
            },
        },
        "pred": {
            D_FIELD.default.name: {
                D_FIELD.POS.name: "VERB",
                # D_FIELD.FEATURES.name:
            }
        },
        "comp": {
            D_FIELD.default.name: {
                D_FIELD.POS.name: "SCONJ",
                # D_FIELD.FEATURES.name:
            }
        },
        "interp": {
            D_FIELD.default.name: {
                D_FIELD.POS.name: "PUNCT"
                # D_FIELD.FEATURES.name:
            }
        },
        "conj": {
            D_FIELD.default.name: {
                D_FIELD.POS.name: "CONJ",
                # D_FIELD.FEATURES.name:
            }
        },
        "adv": {
            D_FIELD.default.name: {
                D_FIELD.POS.name: "ADV",
                # D_FIELD.FEATURES.name:
            }
        },
        "aglt": {
            D_FIELD.special_lemmas.name: [
                (
                    ["być"],  # Kto i co są w subst-ach
                    {
                        D_FIELD.POS.name: "AUX",
                        D_FIELD.FEATURES.name: [
                            ("Mood", "Ind"),
                            ("Tense", "Pres"),
                            ("VerbForm", "Fin"),
                        ],
                    },
                )
            ],
            D_FIELD.default.name: {
                D_FIELD.POS.name: "AUX",
                D_FIELD.FEATURES.name: [
                    ("PronType", "Prs"),
                    ("Reflex", "Yes"),
                    ("Mood", "Ind"),
                    ("Tense", "Pres"),
                    ("VerbForm", "Fin"),
                ],
            },
        },
        "bedzie": {
            D_FIELD.default.name: {
                D_FIELD.POS.name: "AUX",
                # D_FIELD.FEATURES.name:
            },
            D_FIELD.special_words.name: [
                (
                    ["będą", "będzie", "będę", "będziemy", "będziesz"],
                    {
                        D_FIELD.POS.name: "AUX",
                        D_FIELD.FEATURES.name: [
                            ("Tense", "Fut"),
                            ("Mood", "Ind"),
                            ("VerbForm", "Fin"),
                        ],
                    },
                )
            ],
        },
        "burk": {
            D_FIELD.default.name: {
                D_FIELD.POS.name: "NOUN",
                # D_FIELD.FEATURES.name:
            }
        },
        "depr": {
            D_FIELD.default.name: {
                D_FIELD.POS.name: "NOUN",
                # D_FIELD.FEATURES.name:
            }
        },
        "ign": {
            D_FIELD.default.name: {
                D_FIELD.POS.name: "X",
                # D_FIELD.FEATURES.name:
            }
        },
        "dig": {
            D_FIELD.default.name: {
                D_FIELD.POS.name: "NUM",
                # D_FIELD.FEATURES.name:
            }
        },
        "romandig": {
            D_FIELD.default.name: {
                D_FIELD.POS.name: "NUM",
                # D_FIELD.FEATURES.name:
            }
        },
        "siebie": {
            D_FIELD.default.name: {
                D_FIELD.POS.name: "PRON",
                D_FIELD.FEATURES.name: [("PronType", "Prs"), ("Reflex", "Yes")],
            }
        },
        "numcol": {
            D_FIELD.default.name: {
                D_FIELD.POS.name: "NUM",
                D_FIELD.FEATURES.name: [("NumType", "Sets")],
            }
        },
        "winien": {
            D_FIELD.default.name: {D_FIELD.POS.name: "ADJ"},
            D_FIELD.special_lemmas.name: [(["powinien"], {D_FIELD.POS.name: "ADJ", })],
        },
        # 'adj':{
        #     D_FIELD.special_lemmas.name: [
        #         (
        #         ['jaki', 'jakiś', 'żaden', 'wszystek', 'niejaki', 'który', 'taki', 'niektóry', 'którykolwiek', 'któryś',
        #          'ten', 'jakikolwiek', 'tamten', 'każdy', 'wszelki', 'ów'],
        #         {
        #             D_FIELD.POS.name: 'DET',
        #             D_FIELD.FEATURES.name: [('PronType', 'Ind')]
        #         })
        #     ],
        #     D_FIELD.default.name:{
        #         D_FIELD.POS.name:'ADJ'
        #     }
        # },
        "xxx": {D_FIELD.default.name: {D_FIELD.POS.name: "X"}},
        "interj": {D_FIELD.default.name: {D_FIELD.POS.name: "INTJ"}},
        "adj": {
            D_FIELD.special_lemmas.name: [
                (
                    ["wszystek", "wszyscy", "każdy", "wszelki"],
                    {
                        D_FIELD.POS.name: "DET",
                        D_FIELD.FEATURES.name: [("PronType", "Tot")],
                    },
                ),
                (
                    ["jaki", "który"],  # Kto i co są w subst-ach
                    {
                        D_FIELD.POS.name: "DET",
                        D_FIELD.FEATURES.name: [("PronType", "Int,Rel")],
                    },
                ),
                (
                    ["to", "ten", "taki", "tamten", "ów"],
                    {
                        D_FIELD.POS.name: "DET",
                        D_FIELD.FEATURES.name: [("PronType", "Dem")],
                    },
                ),
                (
                    [
                        "jakiś",
                        "kilka",
                        "kilkadziesiąt",
                        "kilkaset",
                        "niektóry",
                        "któryś",
                        "jakikolwiek",
                        "niejaki",
                        "którykolwiek",
                    ],  # kilkanaście jest w num, coś/ktoś w subst
                    {
                        D_FIELD.POS.name: "DET",
                        D_FIELD.FEATURES.name: [("PronType", "Ind")],
                    },
                ),
                (
                    ["żaden"],  # nikt i nic to subst
                    {
                        D_FIELD.POS.name: "DET",
                        D_FIELD.FEATURES.name: [("PronType", "Neg")],
                    },
                ),
            ],
            D_FIELD.default.name: {D_FIELD.POS.name: "ADJ"},
        },
        "adjc": {
            D_FIELD.default.name: {D_FIELD.POS.name: "ADJ"},
            D_FIELD.special_words.name: [
                (
                    [
                        "winien",
                        "gotów",
                        "pewien",
                        "ciekaw",
                        "wart",
                        "pełen",
                        "świadom",
                        "pewnien",
                        "godzien",
                        "łaskaw",
                        "znan",
                        "rad",
                        "wesół",
                        "zdrów",
                    ],
                    {D_FIELD.POS.name: "ADJ"},
                )
            ],
        },
        "qub": {
            D_FIELD.special_lemmas.name: [
                (
                    ["się"],
                    {
                        D_FIELD.POS.name: "PRON",
                        D_FIELD.FEATURES.name: [("PronType", "Prs"), ("Reflex", "Yes")],
                    },
                )
            ],
            D_FIELD.special_words.name: [
                (
                    ["sie", "sia"],
                    {
                        D_FIELD.POS.name: "PRON",
                        D_FIELD.FEATURES.name: [
                            ("PronType", "Prs"),
                            ("Reflex", "Yes"),
                            ("Typo", "Yes"),
                        ],
                    },
                ),
                (
                    ["by"],
                    {
                        D_FIELD.POS.name: "AUX",
                        D_FIELD.FEATURES.name: [
                            ("VerbForm", "Fin"),
                            ("Mood", "Cnd"),
                            ("Aspect", "Imp"),
                        ],
                    },
                ),
            ],
            D_FIELD.default.name: {D_FIELD.POS.name: "PART"},
        },
        "adja": {
            D_FIELD.default.name: {
                D_FIELD.POS.name: "ADJ",
                D_FIELD.FEATURES.name: [("Hyph", "Yes")],
            }
        },
        "prep": {
            D_FIELD.default.name: {
                D_FIELD.POS.name: "ADP",
                D_FIELD.FEATURES.name: [("AdpType", "Prep")],
            }
        },
        "praet": {
            D_FIELD.default.name: {
                D_FIELD.POS.name: "VERB",
                D_FIELD.FEATURES.name: [
                    ("Tense", "Past"),
                    ("VerbForm", "Part"),
                    ("Voice", "Act"),
                ],
            }
        },
        "pact": {
            D_FIELD.default.name: {
                D_FIELD.POS.name: "VERB",
                D_FIELD.FEATURES.name: [
                    ("VerbForm", "Part"),
                    ("Voice", "Act"),
                    ("Tense", "Pres"),
                ],
            }
        },
        "pant": {
            D_FIELD.default.name: {
                D_FIELD.POS.name: "VERB",
                D_FIELD.FEATURES.name: [("Tense", "Past"), ("VerbForm", "Trans")],
            }
        },
        "pcon": {
            D_FIELD.default.name: {
                D_FIELD.POS.name: "VERB",
                D_FIELD.FEATURES.name: [("Tense", "Pres"), ("VerbForm", "Trans")],
            }
        },
        "ppas": {
            D_FIELD.default.name: {
                D_FIELD.POS.name: "VERB",
                D_FIELD.FEATURES.name: [("VerbForm", "Part"), ("Voice", "Pass")],
            }
        },
        "num": {
            D_FIELD.special_lemmas.name: [
                (
                    ["kilkanaście", "kilka", "kilkadziesiąt", "kilkaset"],
                    {
                        D_FIELD.POS.name: "DET",
                        D_FIELD.FEATURES.name: [
                            ("PronType", "Ind"),
                            ("NumType", "Card"),
                        ],
                    },
                )
            ],
            D_FIELD.default.name: {
                D_FIELD.POS.name: "NUM",
                D_FIELD.FEATURES.name: [
                    # ('NumType', 'Sets')
                ],
            },
        },
        "brev": {
            D_FIELD.default.name: {
                D_FIELD.POS.name: "X",
                # D_FIELD.FEATURES.name: [
                #     ('Abbr', 'Yes')
                # ]
            }
        },
        "adjp": {
            D_FIELD.default.name: {
                D_FIELD.POS.name: "ADJ",
                D_FIELD.FEATURES.name: [("PrepCase", "Pre")],
            }
        },
        "fin": {
            D_FIELD.default.name: {
                D_FIELD.POS.name: "VERB",
                D_FIELD.FEATURES.name: {
                    ("VerbForm", "Fin"),
                    ("Tense", "Pres"),
                    ("Mood", "Ind"),
                },
            }
        },
        "ppron12": {
            D_FIELD.default.name: {
                D_FIELD.POS.name: "PRON",
                D_FIELD.FEATURES.name: {("PronType", "Prs")},
            }
        },
        "ppron3": {
            D_FIELD.default.name: {
                D_FIELD.POS.name: "PRON",
                D_FIELD.FEATURES.name: {("PronType", "Prs")},
            }
        },
        "inf": {
            D_FIELD.default.name: {
                D_FIELD.POS.name: "VERB",
                D_FIELD.FEATURES.name: {("VerbForm", "Inf")},
            }
        },
        # 'num':{
        #     #D_FIELD.special_lemmas.name :None,
        #     #D_FIELD.special_words.name :None,
        #     D_FIELD.default.name:{
        #         D_FIELD.POS.name:'NUM'
        #         #,D_FIELD.FEATURES.name:
        #     }
        # },
        "impt": {
            D_FIELD.default.name: {
                D_FIELD.POS.name: "VERB",
                D_FIELD.FEATURES.name: [("Mood", "Imp"), ("VerbForm", "Fin")],
            }
        },
        "imps": {
            D_FIELD.default.name: {
                D_FIELD.POS.name: "VERB",
                D_FIELD.FEATURES.name: [
                    ("Case", "Nom"),
                    ("Gender", "Neut"),
                    ("Negative", "Pos"),
                    ("Number", "Sing"),
                    ("VerbForm", "Part"),
                    ("Voice", "Pass"),
                ],
            }
        },
    },
    D_FIELD.cats.name: {
        "pl": {D_FIELD.default.name: {D_FIELD.FEATURES.name: {("Number", "Plur")}}},
        "pun": {D_FIELD.default.name: {D_FIELD.FEATURES.name: [("Abbr", "Yes")]}},
        "npun": {D_FIELD.default.name: {D_FIELD.FEATURES.name: [("Abbr", "Yes")]}},
        "acc": {D_FIELD.default.name: {D_FIELD.FEATURES.name: {("Case", "Acc")}}},
        "nakc": {D_FIELD.default.name: {D_FIELD.FEATURES.name: {("Variant", "Short")}}},
        "voc": {D_FIELD.default.name: {D_FIELD.FEATURES.name: {("Case", "Voc")}}},
        # 'ppron12': {
        #     D_FIELD.default.name: {
        #         D_FIELD.FEATURES.name: {
        #             ('PronType', 'Prs')
        #         }
        #     }
        # },
        "m1": {
            D_FIELD.default.name: {
                D_FIELD.FEATURES.name: {("Animacy", "Hum"), ("Gender", "Masc")}
            }
        },
        "m2": {
            D_FIELD.default.name: {
                D_FIELD.FEATURES.name: {
                    ("Animacy", "Anim"), ("Gender", "Masc")}
            }
        },
        "m3": {
            D_FIELD.default.name: {
                D_FIELD.FEATURES.name: {
                    ("Animacy", "Inan"), ("Gender", "Masc")}
            }
        },
        "rec": {D_FIELD.default.name: {D_FIELD.FEATURES.name: {}}},
        "nagl": {D_FIELD.default.name: {D_FIELD.FEATURES.name: {}}},
        "agl": {D_FIELD.default.name: {D_FIELD.FEATURES.name: {}}},
        "congr": {D_FIELD.default.name: {D_FIELD.FEATURES.name: {}}},
        "praep": {D_FIELD.default.name: {D_FIELD.FEATURES.name: [("PrepCase", "Pre")]}},
        "_": {D_FIELD.default.name: {D_FIELD.FEATURES.name: {}}},
        "aff": {D_FIELD.default.name: {D_FIELD.FEATURES.name: {("Polarity", "Pos")}}},
        "com": {D_FIELD.default.name: {D_FIELD.FEATURES.name: {("Degree", "Cmp")}}},
        # 'com': {
        #     D_FIELD.default.name: {
        #         D_FIELD.FEATURES.name: {}
        #     }
        # },
        "perf": {D_FIELD.default.name: {D_FIELD.FEATURES.name: {("Aspect", "Perf")}}},
        "imperf": {D_FIELD.default.name: {D_FIELD.FEATURES.name: {("Aspect", "Imp")}}},
        "sg": {D_FIELD.default.name: {D_FIELD.FEATURES.name: {("Number", "Sing")}}},
        "gen": {D_FIELD.default.name: {D_FIELD.FEATURES.name: {("Case", "Gen")}}},
        "nom": {D_FIELD.default.name: {D_FIELD.FEATURES.name: {("Case", "Nom")}}},
        "pos": {D_FIELD.default.name: {D_FIELD.FEATURES.name: {("Degree", "Pos")}}},
        "akc": {D_FIELD.default.name: {D_FIELD.FEATURES.name: {("Variant", "Long")}}},
        "f": {D_FIELD.default.name: {D_FIELD.FEATURES.name: {("Gender", "Fem")}}},
        "dat": {D_FIELD.default.name: {D_FIELD.FEATURES.name: {("Case", "Dat")}}},
        "inst": {D_FIELD.default.name: {D_FIELD.FEATURES.name: {("Case", "Ins")}}},
        "loc": {D_FIELD.default.name: {D_FIELD.FEATURES.name: {("Case", "Loc")}}},
        "neg": {D_FIELD.default.name: {D_FIELD.FEATURES.name: {("Polarity", "Neg")}}},
        "npraep": {
            D_FIELD.default.name: {
                D_FIELD.FEATURES.name: {("PrepCase", "Npr")}}
        },
        "nwok": {D_FIELD.default.name: {D_FIELD.FEATURES.name: {("Variant", "Short")}}},
        "wok": {D_FIELD.default.name: {D_FIELD.FEATURES.name: {("Variant", "Long")}}},
        "vok": {D_FIELD.default.name: {D_FIELD.FEATURES.name: {("Variant", "Long")}}},
        "xxx": {D_FIELD.default.name: {D_FIELD.FEATURES.name: {("Foreign", "Yes")}}},
        "pri": {D_FIELD.default.name: {D_FIELD.FEATURES.name: {("Person", "1")}}},
        "sec": {D_FIELD.default.name: {D_FIELD.FEATURES.name: {("Person", "2")}}},
        "ter": {D_FIELD.default.name: {D_FIELD.FEATURES.name: {("Person", "3")}}},
        "sup": {D_FIELD.default.name: {D_FIELD.FEATURES.name: {("Degree", "Sup")}}},
        "n": {D_FIELD.default.name: {D_FIELD.FEATURES.name: {("Gender", "Neut")}}},
    },
}


def get_main_ud_pos(nkjp_tag):
    main_nkjp_tag = nkjp_tag.split(":")[0]
    try:
        return nkjp_to_ud_dict[D_FIELD.flexemes.name][main_nkjp_tag].get(
            D_FIELD.default.name
        )[D_FIELD.POS.name]
    except:
        return main_nkjp_tag
