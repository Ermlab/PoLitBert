'''Text utils

'''
import sys
import datetime as dt
import os
import mmap

from collections import  namedtuple
import morfeusz2
import nltk


def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines


def create_nltk_sentence_tokenizer():
    
    extra_abbreviations = ['ps',  'inc', 'Corp', 'Ltd', 'Co', 'pkt', 'Dz.Ap', 'Jr', 'jr', 'sp', 'Sp', 'poj',  'pseud', 'krypt', 'sygn', 'Dz.U', 'ws', 'itd', 'np', 'sanskryt', 'nr', 'gł', 'Takht', 'tzw', 't.zw', 'ewan', 'tyt', 'oryg', 't.j', 'vs', 'l.mn', 'l.poj' ]

    position_abbrev = ['Ks', 'Abp', 'abp','bp','dr', 'kard', 'mgr', 'prof', 'zwycz', 'hab', 'arch', 'arch.kraj', 'B.Sc', 'Ph.D', 'lek', 'med', 'n.med', 'bł', 'św', 'hr', 'dziek' ]

    roman_abbrev= [] #['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII', 'XII','XIV','XV','XVI', 'XVII', 'XVIII','XIX', 'XX', 'XXI' ]

    quantity_abbrev = [ 'mln', 'obr./min','km/godz', 'godz', 'egz', 'ha', 'j.m', 'cal', 'obj', 'alk', 'wag' ] # not added: tys.

    actions_abbrev = ['tłum','tlum','zob','wym', 'pot', 'ww', 'ogł', 'wyd', 'min', 'm.i', 'm.in', 'in', 'im','muz','tj', 'dot', 'wsp', 'właść', 'właśc', 'przedr', 'czyt', 'proj', 'dosł', 'hist', 'daw', 'zwł', 'zaw' ]

    place_abbrev = ['Śl', 'płd', 'geogr']

    lang_abbrev = ['jęz','fr','franc', 'ukr', 'ang', 'gr', 'hebr', 'czes', 'pol', 'niem', 'arab', 'egip', 'hiszp', 'jap', 'chin', 'kor', 'tyb', 'wiet', 'sum', 'chor', 'słow', 'węg', 'ros', 'boś']

    military_abbrev = ['kpt', 'kpr', 'obs', 'pil', 'mjr','płk', 'dypl', 'pp', 'gw', 'dyw', 'bryg', 'ppłk', 'mar', 'marsz', 'rez', 'ppor', 'DPanc', 'BPanc', 'DKaw', 'p.uł']

    extra_abbreviations= extra_abbreviations + position_abbrev + roman_abbrev + quantity_abbrev + place_abbrev + actions_abbrev + place_abbrev + lang_abbrev+military_abbrev

    
    # create tokenizer with update abrev
    sentence_tokenizer = nltk.data.load('tokenizers/punkt/polish.pickle')
    #update abbrev
    sentence_tokenizer._params.abbrev_types.update(extra_abbreviations)

    return sentence_tokenizer





InterpTuple = namedtuple("InterpTuple", field_names=["word", "lemat", "tags", "info", "style"])


class MorfAnalyzer(object):


    def __init__(self):
        self._morfeusz = morfeusz2.Morfeusz(separate_numbering=True)

        self._verb_pattern = set(['fin', 'praet', 'inf', 'pred', 'bedzie'])

    def analyse(self,sentence):
        '''Analyse the sentence and return morfeusz2 morf tags
        '''
        analysis = self._morfeusz.analyse(sentence)
        return analysis

    def sentence_valid(self,txt):
        '''Check if the passed txt is valid sentence, should contain min. one verb in proper form'''

        analysis = self.analyse(txt)
                
        # evaluation is done lazy
        return any(InterpTuple(*a[2]).tags.split(":")[0] in self._verb_pattern for a in analysis)


