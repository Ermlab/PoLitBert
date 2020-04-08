#%%
import nltk
import text_utils as tu

import sys
import datetime as dt
import os
from pathlib import Path
from tqdm import tqdm

from collections import namedtuple
import morfeusz2


# import nltk
# #nltk.download('punkt')
# nltk.download()

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


InterpTuple = namedtuple(
    "InterpTuple", field_names=["word", "lemat", "tags", "info", "style"]
)

morf = morfeusz2.Morfeusz(separate_numbering=True)

sentences = [
    "Krzysiek pije piwo o północy",
    "Krzysiek wypije piwo o północy",
    "Krzysiek będzie pił napój o północy",
    "Krzysiek będzie pić piwo o północy",
    "Pić to trzeba umieć",
    "Wolno mi wypić jedno piwo",
    "Winnam wypić piwo",
    "Za oknem słońce",
    "Będzie w domu",
    "Opublikowano w niedzielę",
    "sprawę zgłoszono do Sądu",
    "wniesiono wymaganą opłatę",
    "Obwieszczenie o wydaniu decyzji o ustaleniu lokalizacji inwestycji celu publicznego dla zamierzenia realizowanego w obrębach Serock i Brzeźno (gmina Pruszcz) oraz w obrębach Wudzyn i Wudzynek (gmina Dobrcz)",
    "Abstrakcje Kwiaty Dla dzieci Sport Owoce Człowiek Pojazdy Kuchnia Zwierzęta Martwa Natura Inne Architektura Widoki Drzewa Gory Prostokąty Kwadraty Panoramy Panoramy Slim Tryptyki Mid Tryptyki Tryptyki Wide 4 elementowe regular 4 elementowe 5 elementowe 7 elementowe Tryptyki High 9 elementowe Rosliny Bestsellery Ręcznie malowane!",
    "Wybierz temat Zamówienie kwerendy archiwalnej Zapytanie o papierową dokumentację projektową archiwaliów audiowizualnych Zapytanie o dostępne formaty i jakość kopii cyfrowych Zapytanie o dostępne wersje językowe materiałów audiowizualnych Sugestia zmiany w opisie materiału Inne",
    "UCHWAŁA NR VI/34/15 RADY GMINY BARTNICZKA z dnia 12 czerwca 2015 r. w sprawie określenia zasad udzielania oraz rozliczania dotacji celowych z budżetu Gminy Bartniczka na dofinansowanie realizacji przydomowych",
]


verb_pattern = set(["fin", "praet", "inf", "pred", "bedzie"])

morf_sent = tu.MorfAnalyzer()

for t, s in enumerate(sentences):

    analysis = morf.analyse(s)
    is_valid = any(
        InterpTuple(*a[2]).tags.split(":")[0] in verb_pattern for a in analysis
    )

    is_valid2 = morf_sent.sentence_valid(s)

    print(f"#####\n{t} - {is_valid} {is_valid2}- {s}")

    for i, a in enumerate(analysis):
        print(f"\t{i} {a}")


#%%


text = """
w górach nad jeziorem w lesie nad morzem nad rzeką ---------------------------------------------- Karpaty Wzniesienia Łódzkie Trójmiasto Podkarpacie Równina Radomska Beskid Sądecki Bieszczady Gorce Pieniny Podhale Tatry Tatry Wschodnie Lubelszczyzna Polesie Lubelskie Pogórze Beskidzkie Wyżyna Lubelska Roztocze Tatry Zachodnie Pogórze Śląskie Pogórze Wielickie Małopolska Wyżyna Małopolska Niecka Nidziańska Pogórze Wiśnickie Kotlina Sandomierska Mazowsze Nizina Mazowiecka Puszcza Kampinoska Okolice Warszawy Pogórze Rożnowskie Pogórze Ciężkowickie Wyżyna Krakowsko-Częstochowska Pogórze Strzyżowskie Pogórze Dynowskie Pogórze Przemyskie Jura Krakowsko-Częstochowska Obniżenie Gorlickie Kotlina Jasielsko-Krośnieńska Kotlina Oświęcimska Pogórze Jasielskie Pogórze Bukowskie Płaskowyż Chyrowski Puszcza Niepołomicka Pomorze Brama Krakowska Pomorze Wschodnie Wybrzeże Bałtyckie Pojezierze Kujawskie Wyżyna Przedborska Wyżyna Kielecka Puszcza Sandomierska Dolina Nidy Góry Świętokrzyskie Puszcza Świętorzyska Pomorze Zachodnie Tatry Wysokie Pobrzeże Bałtyckie Roztocze Wschodnie Puszcza Solska Roztocze Zachodnie Roztocze Środkowe Wzniesienia Górowskie Podgórze Bocheńskie Płaskowyż Tarnowski Równina Ornecka Karpaty Zachodnie Karpaty Wschodnie Nizina Sępopolska Pobrzeże Gdańskie Półwysep Helski Mierzeja Helska Mierzeja Wiślana Beskidy Pobrzeże Kaszubskie Kotlina Rabczańska Kotlina Żywiecka Beskid Mały Beskid Makowski Beskid Wyspowy Beskid Żywiecki Kotlina Sądecka Beskid Śląski Równina Warmińska Nizina Nadwiślańska Wybrzeże Staropruskie Wysoczyzna Elbląska Żuławy Wiślane Wybrzeże Koszalińskie Wybrzeże Szczecińskie Nizina Szczecińska Wybrzeże Słowińskie Równina Białogardzka Równina Słupska Wysoczyzna Damnicka Wysoczyzna Żarnowiecka Pradolina Łeby i Redy Uznam i Wolin Wybrzeże Trzebiatowskie Równina Wkrzańska Dolina Dolnej Odry Równina Goleniowska Wzniesienia Szczecińskie Wzgórza Bukowe Równina Wełtyńska Równina Pyrzycko-Stargardzka Równina Nowogardzka Równina Gryficka Mazury Warmia Pojezierze Mazurskie Suwalszczyzna Puszcza Romincka Pojezierze Zachodniosuwalskie Pojezierze Wschodniosuwalskie Pojezierze Olsztyńskie Pojezierze Mrągowskie Kraina Wielkich Jezior Kraina Węgorapy Wzgórza Szeskie Pojezierze Ełckie Równina Mazurska Wielkopolska Nizina Wielkopolska Pojezierze Wielkopolskie Ziemia Lubuska Ziemia Kłodzka Puszcza Notecka Pojezierze Poznańskie Poznański Przełom Warty Pojezierze Chodzieskie Pojezierze Gnieźnieńskie Równina Inowrocławska Równina Wrzesińska Kujawy Podlasie Nizina Podlaska Puszcza Knyszyńska Puszcza Białowieska Puszcza Augustowska Śląsk Nizina Śląska Wyżyna Śląska Pradolina Warciańsko-Odrzańska Bory Stobrawskie Nizina Śląsko-Łużycka Sudety Wysoczyzna Rościsławska Pradolina Wrocławska Równina Wrocławska Dolina Nysy Kłodzkiej Równina Niemodlińska Równina Oleśnicka Równina Opolska Płaskowyż Głubczycki Kotlina Raciborska Bory Dolnośląskie Równina Szprotawska Wysoczyzna Lubińska Równina Legnicka Równina Chojnowska Dolina Środkowej Odry Kotlina Kargowska Kotlina Śremska Przedgórze Sudeckie Pogórze Zachodniosudeckie Sudety Zachodnie Sudety Środkowe Sudety Wschodnie Góry Złote Masyw Śnieżnika Góry Opawskie Brama Lubawska Góry Wałbrzyskie Góry Kamienne Góry Sowie Wyżyna Miechowska Góry Bardzkie Obniżenie Noworudzkie Obniżenie Scinawki Góry Stołowe Pogórze Orlickie Góry Orlickie Góry Bystrzyckie Kotlina Kłodzka Góry Izerskie Góry Kaczawskie Kotlina Jeleniogórska Karkonosze Rudawy Janowickie Pogórze Izerskie Pogórze Kaczawskie Pogórze Wałbrzyskie Pojezierze Pomorskie Pojezierze Zachodniopomorskie Pojezierze Wschodniopomorskie Pojezierze Południowopomorskie Dolina Dolnej Wisły Pojezierze Iławskie Pojezierze Lubuskie Pojezierze Leszczyńskie Lubuski Przełom Odry Pojezierze Łagowskie Równina Torzymska Dolina Kwidzyńska Kotlina Grudziądzka Dolina Fordońska Pojezierze Kaszubskie Pojezierze Starogardzkie Pojezierze Myśliborskie Pojezierze Choszczeńskie Pojezierze Ińskie Wysoczyzna Łobeska Pojezierze Drawskie Wysoczyzna Polanowska Pojezierze Bytowskie Równina Gorzowska Pojezierze Dobiegniewskie Równina Drawska Pojezierze Wałeckie Równina Wałecka Pojezierze Szczecińskie Równina Charzykowska Dolina Gwdy Pojezierze Krajeńskie Bory Tucholskie Dolina Brdy Wysoczyzna Świecka Pojezierze Chełmińskie Pojezierze Brodnickie Dolina Drwęcy Pojezierze Dobrzyńskie Kotlina Gorzowska Kotlina Toruńska Kotlina Płocka Dolina Noteci Kotlina Milicka Beskid Niski Dolina Baryczy Kaszuby Dolny Śląsk Zalew Wiślany Wysoczyzna Rawska Wyżyna Woźnicko-Wieluńska Puszcza Wkrzańska Puszcza Goleniowska Równina Łęczyńsko-Włodawska Puszcza Bukowa Puszcza Drawska Puszcza Gorzowska Puszcza Lubuska Puszcza Karpacka Puszcza Kozienicka Puszcza Pilicka Puszcza Biała Puszcza Bydgoska Puszcza Kurpiowska Puszcza Piska Puszcza Borecka Puszcza Nidzicka Kotlina Szczercowska
"""

text = sentences[12]

sentence_tokenizer = tu.create_nltk_sentence_tokenizer()

ss = sentence_tokenizer.tokenize(text)

morf_sent = tu.MorfAnalyzer()

for t, s in enumerate(ss):
    is_valid2 = morf_sent.sentence_valid(s)
    print(f"#####\n{t} -{is_valid2}- {s}")


InterpTuple = namedtuple(
    "InterpTuple", field_names=["word", "lemat", "tags", "info", "style"]
)

morf = morfeusz2.Morfeusz(separate_numbering=True)
verb_pattern = set(["fin", "praet", "inf", "pred", "bedzie"])
analysis = morf.analyse(text)

for i, a in enumerate(analysis):

    interp = InterpTuple(*a[2])

    if interp.tags.split(":")[0] in verb_pattern:
        print(interp)


#%%
sentences = [
    "Krzysiek pije piwo o północy",
    "Krzysiek wypije piwo o północy",
    "Krzysiek będzie pił napój o północy",
    "Krzysiek będzie pić piwo o północy",
    "Pić to trzeba umieć",
    "Wolno mi wypić jedno piwo",
    "Winnam wypić piwo",
    "Za oknem słońce",
    "Będzie w domu",
    "Opublikowano w niedzielę",
    "sprawę zgłoszono do Sądu",
    "wniesiono wymaganą opłatę",
    "Picie piwa i jeżdżenie na rowerze to fajna rozrywka",
    "Obwieszczenie o wydaniu decyzji o ustaleniu lokalizacji inwestycji celu publicznego dla zamierzenia realizowanego w obrębach Serock i Brzeźno (gmina Pruszcz) oraz w obrębach Wudzyn i Wudzynek (gmina Dobrcz)",
    "Abstrakcje Kwiaty Dla dzieci Sport Owoce Człowiek Pojazdy Kuchnia Zwierzęta Martwa Natura Inne Architektura Widoki Drzewa Gory Prostokąty Kwadraty Panoramy Panoramy Slim Tryptyki Mid Tryptyki Tryptyki Wide 4 elementowe regular 4 elementowe 5 elementowe 7 elementowe Tryptyki High 9 elementowe Rosliny Bestsellery Ręcznie malowane!",
    "Wybierz temat Zamówienie kwerendy archiwalnej Zapytanie o papierową dokumentację projektową archiwaliów audiowizualnych Zapytanie o dostępne formaty i jakość kopii cyfrowych Zapytanie o dostępne wersje językowe materiałów audiowizualnych Sugestia zmiany w opisie materiału Inne",
    "w górach nad jeziorem w lesie nad morzem nad rzeką ---------------------------------------------- Karpaty Wzniesienia Łódzkie Trójmiasto Podkarpacie Równina Radomska Beskid Sądecki Bieszczady Gorce Pieniny Podhale Tatry Tatry Wschodnie Lubelszczyzna Polesie Lubelskie Pogórze Beskidzkie Wyżyna Lubelska Roztocze Tatry Zachodnie Pogórze Śląskie Pogórze Wielickie Małopolska Wyżyna Małopolska Niecka Nidziańska Pogórze Wiśnickie Kotlina Sandomierska Mazowsze Nizina Mazowiecka Puszcza Kampinoska Okolice Warszawy Pogórze Rożnowskie Pogórze Ciężkowickie Wyżyna Krakowsko-Częstochowska Pogórze Strzyżowskie Pogórze Dynowskie Pogórze Przemyskie Jura Krakowsko-Częstochowska Obniżenie Gorlickie Kotlina Jasielsko-Krośnieńska Kotlina Oświęcimska Pogórze Jasielskie Pogórze Bukowskie Płaskowyż Chyrowski Puszcza Niepołomicka Pomorze Brama Krakowska Pomorze Wschodnie Wybrzeże Bałtyckie Pojezierze Kujawskie Wyżyna Przedborska Wyżyna Kielecka Puszcza Sandomierska Dolina Nidy Góry Świętokrzyskie Puszcza Świętorzyska Pomorze Zachodnie Tatry Wysokie Pobrzeże Bałtyckie Roztocze Wschodnie Puszcza Solska Roztocze Zachodnie Roztocze Środkowe Wzniesienia Górowskie Podgórze Bocheńskie Płaskowyż Tarnowski Równina Ornecka Karpaty Zachodnie Karpaty Wschodnie Nizina Sępopolska Pobrzeże Gdańskie Półwysep Helski Mierzeja Helska Mierzeja Wiślana Beskidy Pobrzeże Kaszubskie Kotlina Rabczańska Kotlina Żywiecka Beskid Mały Beskid Makowski Beskid Wyspowy Beskid Żywiecki Kotlina Sądecka Beskid Śląski Równina Warmińska Nizina Nadwiślańska Wybrzeże Staropruskie Wysoczyzna Elbląska Żuławy Wiślane Wybrzeże Koszalińskie Wybrzeże Szczecińskie Nizina Szczecińska Wybrzeże Słowińskie Równina Białogardzka Równina Słupska Wysoczyzna Damnicka Wysoczyzna Żarnowiecka Pradolina Łeby i Redy Uznam i Wolin Wybrzeże Trzebiatowskie Równina Wkrzańska Dolina Dolnej Odry Równina Goleniowska Wzniesienia Szczecińskie Wzgórza Bukowe Równina Wełtyńska Równina Pyrzycko-Stargardzka Równina Nowogardzka Równina Gryficka Mazury Warmia Pojezierze Mazurskie Suwalszczyzna Puszcza Romincka Pojezierze Zachodniosuwalskie Pojezierze Wschodniosuwalskie Pojezierze Olsztyńskie Pojezierze Mrągowskie Kraina Wielkich Jezior Kraina Węgorapy Wzgórza Szeskie Pojezierze Ełckie Równina Mazurska Wielkopolska Nizina Wielkopolska Pojezierze Wielkopolskie Ziemia Lubuska Ziemia Kłodzka Puszcza Notecka Pojezierze Poznańskie Poznański Przełom Warty Pojezierze Chodzieskie Pojezierze Gnieźnieńskie Równina Inowrocławska Równina Wrzesińska Kujawy Podlasie Nizina Podlaska Puszcza Knyszyńska Puszcza Białowieska Puszcza Augustowska Śląsk Nizina Śląska Wyżyna Śląska Pradolina Warciańsko-Odrzańska Bory Stobrawskie Nizina Śląsko-Łużycka Sudety Wysoczyzna Rościsławska Pradolina Wrocławska Równina Wrocławska Dolina Nysy Kłodzkiej Równina Niemodlińska Równina Oleśnicka Równina Opolska Płaskowyż Głubczycki Kotlina Raciborska Bory Dolnośląskie Równina Szprotawska Wysoczyzna Lubińska Równina Legnicka Równina Chojnowska Dolina Środkowej Odry Kotlina Kargowska Kotlina Śremska Przedgórze Sudeckie Pogórze Zachodniosudeckie Sudety Zachodnie Sudety Środkowe Sudety Wschodnie Góry Złote Masyw Śnieżnika Góry Opawskie Brama Lubawska Góry Wałbrzyskie Góry Kamienne Góry Sowie Wyżyna Miechowska Góry Bardzkie Obniżenie Noworudzkie Obniżenie Scinawki Góry Stołowe Pogórze Orlickie Góry Orlickie Góry Bystrzyckie Kotlina Kłodzka Góry Izerskie Góry Kaczawskie Kotlina Jeleniogórska Karkonosze Rudawy Janowickie Pogórze Izerskie Pogórze Kaczawskie Pogórze Wałbrzyskie Pojezierze Pomorskie Pojezierze Zachodniopomorskie Pojezierze Wschodniopomorskie Pojezierze Południowopomorskie Dolina Dolnej Wisły Pojezierze Iławskie Pojezierze Lubuskie Pojezierze Leszczyńskie Lubuski Przełom Odry Pojezierze Łagowskie Równina Torzymska Dolina Kwidzyńska Kotlina Grudziądzka Dolina Fordońska Pojezierze Kaszubskie Pojezierze Starogardzkie Pojezierze Myśliborskie Pojezierze Choszczeńskie Pojezierze Ińskie Wysoczyzna Łobeska Pojezierze Drawskie Wysoczyzna Polanowska Pojezierze Bytowskie Równina Gorzowska Pojezierze Dobiegniewskie Równina Drawska Pojezierze Wałeckie Równina Wałecka Pojezierze Szczecińskie Równina Charzykowska Dolina Gwdy Pojezierze Krajeńskie Bory Tucholskie Dolina Brdy Wysoczyzna Świecka Pojezierze Chełmińskie Pojezierze Brodnickie Dolina Drwęcy Pojezierze Dobrzyńskie Kotlina Gorzowska Kotlina Toruńska Kotlina Płocka Dolina Noteci Kotlina Milicka Beskid Niski Dolina Baryczy Kaszuby Dolny Śląsk Zalew Wiślany Wysoczyzna Rawska Wyżyna Woźnicko-Wieluńska Puszcza Wkrzańska Puszcza Goleniowska Równina Łęczyńsko-Włodawska Puszcza Bukowa Puszcza Drawska Puszcza Gorzowska Puszcza Lubuska Puszcza Karpacka Puszcza Kozienicka Puszcza Pilicka Puszcza Biała Puszcza Bydgoska Puszcza Kurpiowska Puszcza Piska Puszcza Borecka Puszcza Nidzicka Kotlina Szczercowska",
    "UCHWAŁA NR VI/34/15 RADY GMINY BARTNICZKA z dnia 12 czerwca 2015 r. w sprawie określenia zasad udzielania oraz rozliczania dotacji celowych z budżetu Gminy Bartniczka na dofinansowanie realizacji przydomowych",
]


#%%
from flair.data import Sentence
from flair.models import SequenceTagger


tagger = SequenceTagger.load("pos-multi")

#%%

for txt in sentences:
    # txt = sentences[1]
    print(f'\n>>>{txt}')
    sent = Sentence(txt)
    tagger.predict(sent)
    #print(f"\n{sent.to_tagged_string()}")
    for t in sent.tokens:
        print(f'{t}- {t.tags}')
        #sent.tokens[0].get_tag('upos')


# sent.get_language_code()
#%%

# %%
import nltk

# only rus i en

for txt in sentences:
    # txt = sentences[1]
    tokens = nltk.word_tokenize(txt, "polish")
    print(nltk.pos_tag(tokens, lang="pl"))

# %%

# docker run -p 9003:9003 -it djstrong/krnnt:1.0.0

import requests, json
from collections import Counter


url = "http://localhost:9003/?output_format=jsonl"

#get only tag
conv_main_nkjp= lambda x: x[2].split(":")[0]
conv_main_ud = lambda x: tu.get_main_ud_pos(x[2])

for s in sentences:

    # word toknize
    tokens = nltk.word_tokenize(s, "polish")

    print(f"\n>>>{s}")
    token_list = [[tok] for tok in tokens]
    json_data = [[token_list]]

    x = requests.post(url, json=json_data)

    print(x.status_code)
    #print(x.text)
    
    resp = x.json()
    
    list_nkjp_pos= list(map(conv_main_nkjp, resp[0]))
    list_ud_pos= list(map(conv_main_ud, resp[0]))

    stats_nkjp = Counter(list_nkjp_pos)
    stats_ud = Counter(list_ud_pos)
    print(stats_nkjp)


#%%

import requests, json

url = "http://localhost:9003/?output_format=jsonl"



json_data = [[[["ala"], ["ma"], ["kota"], [".", False]],]]

x = requests.post(url, json=json_data)

print(x.status_code)
print(x.text)

resp = x.json()

# %%
