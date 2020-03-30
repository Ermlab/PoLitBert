#%%
import nltk
import text_utils as tu

import sys
import datetime as dt
import os
from pathlib import Path
from tqdm import tqdm 

#%%
from collections import  namedtuple
import morfeusz2

InterpTuple = namedtuple("InterpTuple", field_names=["word", "lemat", "tags", "info", "style"])

morf = morfeusz2.Morfeusz(separate_numbering=True)

# """Operacji tej dokonano w czasie panującego wszechobecnie powojennego chaosu. Zastosowano radykalne lecz efektywne rozwiązanie, do którego zaangażowano DR. Praktycznie jednej nocy z pewnej ilości lokomotyw, wagonów i personelu wydzielono tzw. "Kolonne" (tłum. z niem. jako konwoje), którym powierzono transport sprzętu ze wschodnich Niemiec przez Polskę do granicy radzieckiej.
# """,

sentences = [ 
        'Krzysiek pije piwo o północy',
        'Krzysiek wypije piwo o północy',
        'Krzysiek będzie pił napój o północy',
        'Krzysiek będzie pić piwo o północy',
        'Pić to trzeba umieć',
        'Wolno mi wypić jedno piwo',
        'Winnam wypić piwo',
        "Za oknem słońce",
        "Będzie w domu",
        'Opublikowano w niedzielę',
        'sprawę zgłoszono do Sądu',
        'wniosiono wymaganą opłatę',
        'POLSKA RZECZPOSPOLITA LUDOWA URZĄD PATENTOWY PRL OPIS PATENTOWY Patent dodatkowy do patentu 62944 Zgłoszono: 17.VI.1968 (P 127 544) Pierwszeństwo: Opublikowano: 15.1.1972 64473 KI. 77 a, 65/12 MKP A 63 b, 65/12 UKD 685.639.6 Twórca wynalazku: Józef Danilczyk Właściciel patentu: Politechnika Krakowska, Kraków (Polska)'
]




verb_pattern = set(['fin', 'praet', 'inf', 'pred', 'bedzie'])

morf_sent = tu.MorfAnalyzer()

for t, s in enumerate(sentences):

    analysis = morf.analyse(s)
    is_valid = any(InterpTuple(*a[2]).tags.split(":")[0] in verb_pattern for a in analysis)

    is_valid2 = morf_sent.sentence_valid(s)

    print(f'#####\n{t} - {is_valid} {is_valid2}- {s}')

    

    # for i,a in enumerate(analysis):
    #     print(f'\t{i} {a}')

    
#%% 

text = '''
POLSKA RZECZPOSPOLITA LUDOWA URZĄD PATENTOWY PRL OPIS PATENTOWY Patent dodatkowy do patentu 62944 Zgłoszono: 17.VI.1968 (P 127 544) Pierwszeństwo: Opublikowano: 15.1.1972 64473 KI. 77 a, 65/12 MKP A 63 b, 65/12 UKD 685.639.6 Twórca wynalazku: Józef Danilczyk Właściciel patentu: Politechnika Krakowska, Kraków (Polska) Przyrząd do gry w siatkówkę Przedmiotem wynalazku jest przyrząd do nauki i doskonalenia zbijania piłki przy grze w siat kówkę. Znany i stosowany jest przyrząd do nauki i do skonalenia ataku przy grze w siatkówkę na przy kład według patentu nr 62944, którego istota po lega na tym, że do korpusu przyrządu przymoco wane jest wahliwie za pośrednictwem przegubu ramię wyrzucające piłkę, która w czasie przebie gania po torze wyrzutu wykorzystana jest do zbicia. Przyrząd ten jednak mimo niewątpliwego usprawnienia ćwiczeń zbijania piłki ma szereg istotnych niedogodności. Niedogodności te polegają na tym, że w korpusie urządzenia można umiesz czać tylko jedną piłkę, urządzenie pracuje bardzo głośno, co wpływa denerwująco na ćwiczącego, na stawianie kąta ramienia odbywa się poprzez śrubę zaciskową i ręczne podnoszenie lub opuszczanie ramienia, która to czynność jest uciążliwa dla obsługującego. Niezależnie od tego, na skutek sta łego korpusu przyrządu nie można go wydłużać lub skręcać w płaszczyźnie pionowej, a tym sa mym dopasowywać do wzrostu obsługującego. W przeciwieństwie do tego w przyrządzie we dług wynalazku w specjalnym koszu umieszcza się kilka piłek, które kolejno automatycznie zostają podane do wyrzutni, urządzenie pracuje cicho ze względu na zastosowanie w nim specjalnego zde rzaka, który siłę uderzenia ramienia wyrzutu 10 15 20 przenosi na podstawę przyrządu. Ustawienie kąta pionowego ramienia wyrzutu, a tym samym kąta wyrzutu, jest regulowane specjalną zębatką z ko łem zębatym, którego obrót powoduje szybką i łatwą zmianę kąta nachylenia. Również łatwe jest przystosowanie przyrządu do wysokości obsługują cego poprzez skręcanie lub wydłużanie korpusu przyrządu.
'''    

text='''Wydarzenia i nowości Konstrukcje Publikacje Producenci Dachy skośne Publikacje Producenci Dachy płaskie Publikacje Producenci Pokrycia dachowe Pokrycia ceramiczne Publikacje Producenci Pokrycia cementowe Publikacje Producenci Pokrycia blaszane Publikacje Producenci Papy Publikacje Producenci Gonty bitumiczne Publikacje Producenci Łupek Publikacje Producenci Płyty dachowe Publikacje Producenci Inne Publikacje Producenci Dachy zielone Publikacje Producenci Dachy odwrócone Publikacje Producenci Okno w dachu Publikacje Producenci Folie dachowe Publikacje Producenci Ocieplenia dachów skośnych Publikacje Producenci Ocieplenia dachów płaskich Publikacje Producenci Akcesoria dachowe Publikacje Producenci Kominy Publikacje Producenci Rynny i odwodnienia Publikacje Producenci Ochrona odgromowa Publikacje Producenci Renowacja Publikacje Producenci Chemia budowlana Publikacje Producenci Maszyny i narzędzia Publikacje Producenci Obróbki blacharskie Publikacje Producenci Poddasza Publikacje Producenci Wentylacja dachów Publikacje Producenci Dom energooszczędny Publikacje Producenci Proekologiczne budowanie Publikacje Producenci Instrukcje Poradnik Publikacje Producenci Dylematy Inne TV Dachy Forum szkół Dla dekarzy Z życia PSD Szkolenia Budownictwo w statystykach BHP na budowie Rzeczoznawcy Organizacje branżowe Targi Wydawnictwa Konkursy i szkolenia Kontakt.
Woda i wilgoć, które mogą przenikać do wnętrza przegród nawet w obliczu niewielkich opadów deszczu i śniegu, stanowią jedno z największych zagrożeń stabilności i wytrzymałości konstrukcji dachów płaskich. Problem objawia się najczęściej w sezonie jesienno-zimowym i dotyczy głównie obiektów wielkopowierzchniowych, takich jak magazyny, hale produkcyjne czy centra logistyczne. O czym powinni pamiętać inżynierowie, projektując bezpieczny dach płaski?
Dach z zerowym kątem nachylenia to gwarancja problemów eksploatacyjnych: mechanicznej degradacji materiału izolacyjnego, korozji stalowych blach i łączników mechanicznych, zmniejszonej efektywności energetycznej obiektu. Dlatego też warunkiem koniecznym jest uwzględnienie odpowiednich spadków dachu. W Polsce, w sezonie jesienno-zimowym okres zalegania śniegu może sięgać nawet kilku miesięcy. Woda, która przez ten czas nie jest dostatecznie szybko usuwana mechanicznie lub poprzez odparowanie, stanowi prawdziwą próbę zarówno dla szczelności, jak i wytrzymałości mechanicznej konstrukcji.
Jak podkreśla Adam Buszko, ekspert firmy Paroc, nawet nieduże z pozoru błędy mogą o sobie szybko przypomnieć w postaci poważnych przecieków. – Nagromadzona wilgoć może dochodzić nawet do 10-20 milimetrów na metr kwadratowy, co odpowiada 10-20% objętości izolacji w zależności od jej grubości – wyjaśnia. Ryzyko problemów wzrasta zwłaszcza w okresie niskich temperatur, kiedy woda penetruje wszelkie szczeliny i ewentualne rozwarstwienia. – Cykle zamarzania i rozmarzania mogą prowadzić do powstawania nieszczelności w warstwie hydroizolacji oraz na jej połączeniach z innymi konstrukcjami – na przykład ścianami elewacji – dodaje ekspert Paroc.
W przypadku dachów płaskich planowane spadki powinny wynosić minimum 2-3°. W wyjątkowych sytuacjach, gdy ze względów konstrukcyjnych spadki muszą wynosić mniej niż 2° (np. w zlewniach pogłębionych), należy podjąć odpowiednie działania w celu ograniczenia ryzyka wystąpienia zatoisk wody. Warstwa hydroizolacji powinna wówczas składać się z trzech warstw grubych, zbrojonych, odpornych na niskie temperatury pap termozgrzewalnych. Kluczowe dla bezpieczeństwa dachu płaskiego jest też zastosowanie odpowiedniego materiału izolacyjnego.
W przypadku montażu sztywnych płyt styropianowych, nawet przy słabym wietrze wzrasta ryzyko albo niedogrzania połączenia na zakładach, albo w drugą stronę – do stopienia styropianu. Problem ten dość mocno rzutuje na zachowanie się wody na gotowym dachu, a często zależy tylko w minimalnym stopniu od umiejętności oraz doświadczenia wykonawcy. W przypadku zimnego zgrzewu, im mniejsze zachowamy spadki, tym większe ryzyko penetracji szpar pomiędzy warstwami papy przez wodę. Jeśli dojdzie zaś do wytopienia styropianu, w warstwie pokrycia wytworzą się zagłębienia, w których stać będzie woda, a papa na zakładach podlegać będzie intensywniejszym cyklom naprężeń. Z powyższych względów najlepiej stosować izolacje niepalne, takie jak wełna kamienna, w przypadku której nie występuje ryzyko stopienia materiału.
'''
text='''Przedsiębiorstwo Badawczo-Wdrożeniowe Acrylmed dr Ludwika Własińska Sp. z o.o. 63-100 Śrem, ul. Mickiewicza 33.
Sąd Rejonowy Poznań-Nowe Miasto i Wilda w Poznaniu, IX Wydział Gospodarczy Krajowego Rejestru Sądowego.
Na naszych stronach internetowych stosujemy pliki cookies. Korzystając z naszych serwisów internetowych bez zmiany ustawień przeglądarki wyrażasz zgodę na stosowanie plików cookies zgodnie z Polityką prywatności.
BWW Law & Tax doradzała międzynarodowemu koncernowi budowlanemu Strabag w transakcji nabycia projektu biurowo-handlowego w Warszawie
Zapraszamy do lektury komentarza Jarosława Ziółkowskiego dot. PIT a zbycia praw i obowiązków komandytariuszy w spółce komandytowej.
Abstrakcje Kwiaty Dla dzieci Sport Owoce Człowiek Pojazdy Kuchnia Zwierzęta Martwa Natura Inne Architektura Widoki Drzewa Gory Prostokąty Kwadraty Panoramy Panoramy Slim Tryptyki Mid Tryptyki Tryptyki Wide 4 elementowe regular 4 elementowe 5 elementowe 7 elementowe Tryptyki High 9 elementowe Rosliny Bestsellery Ręcznie malowane!
to aktywna i dynamicznie rozwijająca się firma rodzinna o ugruntowanej pozycji na rynku. Tworzy ją zespół kreatywnych, energicznych i ambitnych osób, które wiedzą jak sprostać zadaniu aby osiągnąć sukces. .
Nasza strona internetowa używa plików cookies (tzw. ciasteczka) w celach statystycznych, reklamowych oraz funkcjonalnych. Dzięki nim możemy indywidualnie dostosować stronę do twoich potrzeb. Każdy może zaakceptować pliki cookies albo ma możliwość wyłączenia ich w przeglądarce, dzięki czemu nie będą zbierane żadne informacje. Regulamin strony.
W wyniku nowelizacji ustawy Prawo zamówień publicznych, która weszła w życie w dniu 28 lipcu 2016 roku, dwie spośród jedenastu przesłanek...
Z początkiem 2017 r. zaczęły obowiązywać nowe regulacje dotyczące rejestracji podatników dla celów podatku od towarów i usług. Znowelizowane przepisy służyć mają realizacji założeń Ministerstwa Rozwoju i Finansów dotyczących uszczelniania systemu podatkowego. W praktyce przysparzają one kłopotów nie tylko nowopowstałym podmiotom.
Nowelizacją ustawy o podatku od towarów i usług ustawodawca wyposażył organy podatkowe w nowe uprawnienia. Ma to m.in. związek z podziałem dotychczasowego rejestru VAT na dwa „podrejestry” – rejestr VAT czynny i zwolniony. Po przeprowadzeniu weryfikacji (czynności sprawdzających) naczelnik urzędu skarbowego jest obecnie uprawniony do rejestracji podatnika jako „podatnika VAT czynnego” lub „podatnika VAT zwolnionego”. Może też w ogóle nie dokonać jego rejestracji bez konieczności zawiadomienia o tym przedsiębiorcy, jeżeli np. mimo udokumentowanych prób nie ma możliwości skontaktowania się z podatnikiem, podmiot nie istnieje lub podał w zgłoszeniu rejestracyjnym dane niezgodne z prawdą.
"Sprzedaż posiłków a stawka VAT" - komentarz Urszuli Mazurek do artykułu red. Aleksandry Tarki w "Rzeczpospolitej"
'''

sentence_tokenizer = tu.create_nltk_sentence_tokenizer()

ss = sentence_tokenizer.tokenize(text)

for t, s in enumerate(ss):

    is_valid2 = morf_sent.sentence_valid(s)

    print(f'#####\n{t} -{is_valid2}- {s}')


#%%

CorpusProcessingTuple = namedtuple("CorpusProcessingTuple", 
        field_names=["file_path", "split_each_line_as_doc", "check_valid_sentence"])

files_to_proces = [
    CorpusProcessingTuple('./data/corpus_wikipedia_2020-02-01.txt', False, False),
    CorpusProcessingTuple('./data/corpus_oscar.txt', True, True ),
    CorpusProcessingTuple('./data/corpus_books.txt', False, False ),
    CorpusProcessingTuple('./data/corpus_subtitles.txt', False, False ),
    CorpusProcessingTuple('./data/corpus_patents.txt', False, True ),
]


files_to_proces = [
    CorpusProcessingTuple('./data/corpus_raw/corpus_govtech_small.txt', False, True),
    CorpusProcessingTuple('./data/corpus_raw/corpus_oscar_100k.txt', True, True ),
]


#%%

input_file = './data/corpus_raw/corpus_govtech_small.txt'


p = Path(input_file)
output_path = f"{p.with_suffix('')}_lines.txt"


sentence_tokenizer = tu.create_nltk_sentence_tokenizer()
total_lines = tu.get_num_lines(input_file)

#%%

for corpus_tuple in files_to_proces:
    
    input_file = corpus_tuple.file_path

    split_each_line_as_doc = corpus_tuple.split_each_line_as_doc 

    check_valid_sentence= corpus_tuple.check_valid_sentence


    print(input_file)
    p = Path(input_file)
    output_path = f"{p.with_suffix('')}_lines.txt"

    print(f"in file={input_file}\nout file={output_path}")

    t0=dt.datetime.now()

    total_lines = get_num_lines(input_file)


    with open(output_path, 'w+') as output_file:
            with open(input_file) as f:
                i=0
                text=''
                for line in tqdm(f,total=total_lines):

                    # get block of text to new line which splits ariticles
                    text+=line

                    i+=1
                    if split_each_line_as_doc or line.strip() == '' or i%100==0:
                        #if split_each_line_as_doc is set then add new line after each line, if not then read file in block of 100 lines up to empty line and process lines

                        sentences = sentence_tokenizer.tokenize(text)

                        file_content = ''
                        for sentence in sentences:

                            sentence = sentence.strip()
                            if check_valid_sentence and not sentence_valid(sentence):
                                # omit sentence if is not valid
                                continue

                            file_content += sentence
                            file_content+='\n'
                        output_file.write(file_content)
                        
                        output_file.write('\n')
                        text=''

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
                        

            