#%%
import nltk
import text_utils as tu

import sys
import datetime as dt
import os
from pathlib import Path
from tqdm import tqdm

from collections import namedtuple, Counter
import morfeusz2
import stanza


# import nltk
# #nltk.download('punkt')
# nltk.download()
# only rus i en

# for txt in sentences:
#     # txt = sentences[1]
#     tokens = nltk.word_tokenize(txt, "polish")
#     print(nltk.pos_tag(tokens, lang="pl"))

InterpTuple = namedtuple(
    "InterpTuple", field_names=["word", "lemat", "tags", "info", "style"]
)

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
    "Wydarzenia i nowości Konstrukcje Publikacje Producenci Dachy skośne Publikacje Producenci Dachy płaskie Publikacje Producenci Pokrycia dachowe Pokrycia ceramiczne Publikacje Producenci Pokrycia cementowe Publikacje Producenci Pokrycia blaszane Publikacje Producenci Papy Publikacje Producenci Gonty bitumiczne Publikacje Producenci Łupek Publikacje Producenci Płyty dachowe Publikacje Producenci Inne Publikacje Producenci Dachy zielone Publikacje Producenci Dachy odwrócone Publikacje Producenci Okno w dachu Publikacje Producenci Folie dachowe Publikacje Producenci Ocieplenia dachów skośnych Publikacje Producenci Ocieplenia dachów płaskich Publikacje Producenci Akcesoria dachowe Publikacje Producenci Kominy Publikacje Producenci Rynny i odwodnienia Publikacje Producenci Ochrona odgromowa Publikacje Producenci Renowacja Publikacje Producenci Chemia budowlana Publikacje Producenci Maszyny i narzędzia Publikacje Producenci Obróbki blacharskie Publikacje Producenci Poddasza Publikacje Producenci Wentylacja dachów Publikacje Producenci Dom energooszczędny Publikacje Producenci Proekologiczne budowanie Publikacje Producenci Instrukcje Poradnik Publikacje Producenci Dylematy Inne TV Dachy Forum szkół Dla dekarzy Z życia PSD Szkolenia Budownictwo w statystykach BHP na budowie Rzeczoznawcy Organizacje branżowe Targi Wydawnictwa Konkursy i szkolenia Kontakt",
    "Woda i wilgoć, które mogą przenikać do wnętrza przegród nawet w obliczu niewielkich opadów deszczu i śniegu, stanowią jedno z największych zagrożeń stabilności i wytrzymałości konstrukcji dachów płaskich.",
    "Problem objawia się najczęściej w sezonie jesienno-zimowym i dotyczy głównie obiektów wielkopowierzchniowych, takich jak magazyny, hale produkcyjne czy centra logistyczne.",
    "O czym powinni pamiętać inżynierowie, projektując bezpieczny dach płaski?",
    "Dach z zerowym kątem nachylenia to gwarancja problemów eksploatacyjnych: mechanicznej degradacji materiału izolacyjnego, korozji stalowych blach i łączników mechanicznych, zmniejszonej efektywności energetycznej obiektu.",
    "Dlatego też warunkiem koniecznym jest uwzględnienie odpowiednich spadków dachu.",
    "W Polsce, w sezonie jesienno-zimowym okres zalegania śniegu może sięgać nawet kilku miesięcy.",
    "Woda, która przez ten czas nie jest dostatecznie szybko usuwana mechanicznie lub poprzez odparowanie, stanowi prawdziwą próbę zarówno dla szczelności, jak i wytrzymałości mechanicznej konstrukcji.",
    "Jak podkreśla Adam Buszko, ekspert firmy Paroc, nawet nieduże z pozoru błędy mogą o sobie szybko przypomnieć w postaci poważnych przecieków.",
    "– Nagromadzona wilgoć może dochodzić nawet do 10-20 milimetrów na metr kwadratowy, co odpowiada 10-20% objętości izolacji w zależności od jej grubości – wyjaśnia.",
    "Ryzyko problemów wzrasta zwłaszcza w okresie niskich temperatur, kiedy woda penetruje wszelkie szczeliny i ewentualne rozwarstwienia.",
    "– Cykle zamarzania i rozmarzania mogą prowadzić do powstawania nieszczelności w warstwie hydroizolacji oraz na jej połączeniach z innymi konstrukcjami – na przykład ścianami elewacji – dodaje ekspert Paroc.",
    "W przypadku dachów płaskich planowane spadki powinny wynosić minimum 2-3°.",
    "W wyjątkowych sytuacjach, gdy ze względów konstrukcyjnych spadki muszą wynosić mniej niż 2° (np. w zlewniach pogłębionych), należy podjąć odpowiednie działania w celu ograniczenia ryzyka wystąpienia zatoisk wody.",
    "W przypadku montażu sztywnych płyt styropianowych, nawet przy słabym wietrze wzrasta ryzyko albo niedogrzania połączenia na zakładach, albo w drugą stronę – do stopienia styropianu.",
    "Problem ten dość mocno rzutuje na zachowanie się wody na gotowym dachu, a często zależy tylko w minimalnym stopniu od umiejętności oraz doświadczenia wykonawcy.",
    "W przypadku zimnego zgrzewu, im mniejsze zachowamy spadki, tym większe ryzyko penetracji szpar pomiędzy warstwami papy przez wodę.",
    "Jeśli dojdzie zaś do wytopienia styropianu, w warstwie pokrycia wytworzą się zagłębienia, w których stać będzie woda, a papa na zakładach podlegać będzie intensywniejszym cyklom naprężeń.",
    "Z powyższych względów najlepiej stosować izolacje niepalne, takie jak wełna kamienna, w przypadku której nie występuje ryzyko stopienia materiału.",
    "Ze względu na naturalną paroprzepuszczalność oraz odporność na wnikanie wilgoci, produkty z wełny kamiennej sprawdzają się zarówno w przypadku konstrukcji dachów płaskich wentylowanych, jak i niewentylowanych.",
    "W pierwszym przypadku stosować można izolacyjne płyty dwuwarstwowe z wierzchnią warstwą utwardzoną lub dwie płyty: jedną twardą – jako płytę wierzchnią, drugą miękką i lżejszą – jako spodnią.",
    "Podobne rozwiązanie sprawdza się także w przypadku konstrukcji wentylowanych.",
    "Gotowe rozwiązania w tej dziedzinie dostarczają producenci izolacji.",
    "– W systemie PAROC Air spodnia płyta izolacyjna wyposażona jest w system rowków, które umożliwiają sprawny transport pary wodnej w kierunku wylotów – tłumaczy Adam Buszko.",
    "– Wierzchni arkusz z wełny kamiennej został zaś opracowany tak, by zapewniać trwałe, twarde i ognioodporne podłoże dla większości typów płaskich pokryć dachowych, a także dla izolacji warstwy nośnej w miejscach remontów – dodaje.",
    "Odpowiednio dobrana izolacja, w połączeniu właściwym rozmieszczeniem spadków i odwodnień, pozwala na sprawne odprowadzenie wilgoci z konstrukcji przez cały okres jej eksploatacji.",
    "Za optymalne przyjmuje się, że osuszanie połaci dachu na poziomie 0,5 kg wody/m2 na dobę, co skutecznie eliminuje zagrożenie gromadzenia się wilgoci – również na etapie budowy.",
    "Przedsiębiorstwo Badawczo-Wdrożeniowe Acrylmed dr Ludwika Własińska Sp. z o.o. 63-100 Śrem, ul. Mickiewicza 33",
    "Sąd Rejonowy Poznań-Nowe Miasto i Wilda w Poznaniu, IX Wydział Gospodarczy Krajowego Rejestru Sądowego",
    "Na naszych stronach internetowych stosujemy pliki cookies.",
    "Korzystając z naszych serwisów internetowych bez zmiany ustawień przeglądarki wyrażasz zgodę na stosowanie plików cookies zgodnie z Polityką prywatności.",
    "Na mocy uchwały Rady Powiatu Głogowskiego z dnia 28 marca 2007 r. Nr VI/54/2007 został utworzony zespół publicznych placówek kształcenia ustawicznego i praktycznego o nazwie Głogowskie Centrum Edukacji Zawodowej w Głogowie przy ulicy Piaskowej 1.",
    "Centrum Kształcenia Ustawicznego w Głogowie - ustawiczne kształcenie, dokształcanie i doskonalenie osób dorosłych.",
    "- dostrzeganie swojej niepowtarzalności, a także niepowtarzalności innych, szanowanie jej.",
    "- poznanie różnych środków transportu: lądowego, wodnego, powietrznego.",
    "- rozpoznawanie drzew po ich liściach i owocach, zbieranie owoców drzew, wzbogacanie nimi kącika przyrody: wykorzystywanie owoców w działalności plastycznej, technicznej, matematycznej, muzycznej oraz w inny, niestandardowy sposób.",
    "Czytanie bajki Kubuś Puchatek przez Wójta Gminy Wiśniew p. Krzysztofa Kryszczuka.",
    "PLAN PRACY DYDAKTYCZNEJ NA WRZESIEŃ 2016 W GRUPIE 6-latków Tygryski Tydzień I Przedszkole drugi dom Tydzień III Uliczne sygnały Tydzień II Przedszkole drugi dom Tydzień IV Jesień w lesie Treści programowe",
    "Dell OptiPlex 9020 Konfiguracja i funkcje komputera Informacja o ostrzeżeniach PRZESTROGA: Napis OSTRZEŻENIE informuje o sytuacjach, w których występuje ryzyko uszkodzenia sprzętu, obrażeń ciała lub śmierci.",
    "Zasadnicze cele konkursu to m.in. kształtowanie postaw patriotycznych młodzieży poprzez propagowanie i pogłębianie wiedzy o organizacji i działalności Służby Zwycięstwu Polski-Związku Walki Zbrojnej – Armii Krajowej oraz formacji poakowskich na terenie zamojskiego Inspektoratu Armii Krajowej, kultywowanie wartości, ideałów i postaw żołnierzy Polskiego Państwa Podziemnego, zainspirowanie i zachęcanie młodzieży do podjęcia samodzielnych badań nad historią swojej rodziny, środowiska związanego z miejscem zamieszkania",
    "Z kolei w imieniu organizatora konkursu głos zabrał Prezes ŚZŻAK Okręg Zamość Poseł Sławomir Zawiślak, który przedstawiając efekty tegorocznej, już IX edycji konkursu podziękował za pomoc w jego organizacji Patronom Honorowym, Dyrektorom szkół, nauczycielom, członkom związku w tym Weteranom AK – wszystkim tym, którzy od lat popierają inicjatywę Związku i przyczyniają się do uświetnienia konkursu.",
    "Na czwartkowym (13.06) kongresie pojawili się Marta Niewczas, podkarpacki pełnomocnik Europy Plus, poseł Janusz Palikot, europoseł Marek Siwiec, Robert Smucz, przewodniczący zarządu okręgu rzeszowsko-tarnobrzeskiego RP i wiceszef Europy Plus na Podkarpaciu oraz jeden z liderów stowarzyszenia Ordynacka Robert Kwiatkowski.",
    "ochrona ppoż-znaki uzupełniające prom.elektromag.-znaki ostrzeg.",
    "subst.chem.-znaki kateg.niebezp.",
    "taśma odradzająca z folii PE telefony alarmowe-tablice urz.elektryczne-znaki informac.",
    "urz.elektryczne-znaki ostrzeg.",
    "urz.elektryczne-znaki zakazu pozostałe znaki i tablice",
    "Wyświetl posty z ostatnich: Wszystkie Posty1 Dzień7 Dni2 Tygodnie1 Miesiąc3 Miesiące6 Miesięcy1 Rok Najpierw StarszeNajpierw Nowsze Zobacz poprzedni temat : Zobacz następny temat",
    "Na skróty:o nasatakiofftopz poza sojuszufunna wasze zyczenie zakladamy tematynie moge sie zarejestrowaccarna listaregulaminpakty",
    "3.1) Uprawnienia do wykonywania określonej działalności lub czynności, jeżeli przepisy prawa nakładają obowiązek ich posiadania Zamawiający uzna warunek za spełniony, jeżeli Wykonawca wykaże, iż posiada zezwolenie na wykonywanie działalności ubezpieczeniowej, o którym mowa w Ustawie z dnia 22 maja 2003 r. o działalności ubezpieczeniowej (tekst jednolity Dz. U. z 2013 r. poz. 950 z późn. zm.), a w przypadku gdy rozpoczął on działalność przed wejściem w życie Ustawy z dnia 28 lipca 1990 r. o działalności ubezpieczeniowej (Dz. U. Nr 59, poz. 344 ze zm.) zaświadczenie Ministra Finansów o posiadaniu zgody na wykonywanie działalności ubezpieczeniowej III.3.2) Wiedza i doświadczenie III.3.3) Potencjał techniczny III.3.4) Osoby zdolne do wykonania zamówienia",
    "Volvo Ocean Race (Wcześniej Whitbread Round the World) jest dla żeglarzy tym, czym dla wielbicieli motosportu 24-godzinny wyścig LeMans, a dla alpinistów zimowe ataki szczytowe.",
    "W 1973 r. na starcie Whitbread Round the World załogę Copernicusa stanowili: kapitan Zygfryd Perlicki, Zbigniew Puchalski, Bogdan Bogdziński, Ryszard Mackiewicz i Bronisław Tarnacki.",
    "Możliwość rozliczenia mieszkania w cenie i dopłaty reszty kwoty,",
    "Obecna na rynku od roku 1976, amerykańska firma ASP (Armament Systems and Procedures, Inc.) to prekursor, a obecnie też lider produkcji najwyższej jakości akcesoriów dla służb mundurowych.",
    "przyjmowania oświadczeń odstąpienia od zawartych umów sprzedaży na odległość, zgodnie z postanowieniami niniejszego Regulaminu oraz przepisami Rozdziału 4 ustawy z dnia 30 maja 2014 r. o prawach konsumenta, co stanowi prawnie uzasadniony interes Sprzedawcy (podstawa prawna przetwarzania danych: art. 6 ust. 1 lit. f RODO),",
    "Studenci kierunku Architektura Krajobrazu otrzymują przygotowanie z zakresu nauk przyrodniczych, rolniczych, technicznych i sztuk pięknych oraz umiejętności wykorzystania jej w pracy zawodowej z zachowaniem zasad prawnych i estetycznych.",
    "Poza tym wydział ten zajmuje się sprawami związanymi z realizacją zadań inwestycyjnych i remontowych, przygotowaniem i prowadzeniem procedur przetargowych na realizacją inwestycji i remontów, podejmowaniem działań w celu pozyskiwania zewnętrznych źródeł finansowania działalności inwestycyjnej, w tym opracowywaniem stosownych wniosków oraz koordynowaniem spraw związanych ze sprawozdawczością i rozliczaniem inwestycji.",
    "Biuro Promocji, Informacji i Rozwoju Powiatu prowadzi sprawy związane z promocją i rozwojem powiatu, współpracą z mediami, planowaniem strategicznym oraz koordynowaniem działań związanych ze współpracą zagraniczną.",
    "Do głównych zadań Biura Kadr i Płac należy opracowywanie zasad polityki kadrowej i zarządzanie kadrami oraz sporządzanie wykazów etatów i planów rozmieszczenia pracowników.",

    

]

#%% test with morfeusz2

morf_sent = tu.MorfeuszAnalyzer()

morf_stanza = tu.StanzaAnalyzer()

morf_krnnt = tu.KRNNTAnalyzer()

for t, s in enumerate(sentences):

    is_valid1 = morf_sent.sentence_valid(s)

    is_valid2 = morf_stanza.sentence_valid(s)

    is_valid3 = morf_krnnt.sentence_valid(s)

    print(f"#####\n{s}\n morfeusz={is_valid1} stanza={is_valid2} krnnt={is_valid3}")



#%% pos with flair
from flair.data import Sentence
from flair.models import SequenceTagger


tagger = SequenceTagger.load("pos-multi")

#%%


sentence = sentences[0]
print(f"\n>>>{sentence}")
sent = Sentence(sentence)
tagger.predict(sent)
print(f"\n{sent.to_tagged_string()}")
for t in sent.tokens:
    print(f"{t}- {t.get_tag('upos').value} {t.get_tag('upos').score}")


conv_flair_get_pos = lambda x: x.get_tag("upos").value
flair_ud_pos = list(map(conv_flair_get_pos, sent.tokens))
stats_flair_pos = Counter(flair_ud_pos)

print(stats_flair_pos)


# %% sentence taggers

# docker run -p 9003:9003 -it djstrong/krnnt:1.0.0


import itertools

flatten = itertools.chain.from_iterable
# map words in sentence to list of pos
conv_stanza_pos = lambda x: [w.pos for w in x.words]
conv_stanza_xpos = lambda x: [w.xpos for w in x.words]


stanza.download("pl")
nlp = stanza.Pipeline(
    "pl", processors="tokenize,pos,lemma", verbose=False
)  # initialize neural pipeline

import requests, json

url = "http://localhost:9003/?output_format=jsonl"
# url = "http://localhost:9003/?input_format=lines&output_format=tsv"


# get only tag
conv_main_nkjp = lambda x: x[2].split(":")[0]
conv_main_ud = lambda x: tu.get_main_ud_pos(x[2])

for s in sentences[0:]:

    print(f"\n>>>{s}\n sent len={len(s)}")
    #run krnnt tagger
    x = requests.post(url, data=s.encode("utf-8"))
    # print(x.status_code)
    # print(x.text)

    resp = x.json()
    list_nkjp_pos = list(map(conv_main_nkjp, resp[0]))
    krnnt_pos = list(map(conv_main_ud, resp[0]))

    stats_nkjp = Counter(list_nkjp_pos)
    stats_krnnt_ud = Counter(krnnt_pos)
    # print(f"NKJP tags stats={stats_nkjp}")
    print(f"krnnt UD   tags stats={stats_krnnt_ud}")
    print(f'ud sequence={",".join(krnnt_pos)}')

    # run stanza tagger
    doc = nlp(s)  # run annotation over a sentence

    # flatten if found many sentences
    stanza_pos = list(flatten(map(conv_stanza_pos, doc.sentences)))
    stanza_xpos = list(flatten(map(conv_stanza_xpos, doc.sentences)))
    stats_stanza_pos = Counter(stanza_pos)
    stats_stanza_xpos = Counter(stanza_xpos)
    print(f"stanza UD   tags stats={stats_stanza_pos}")
    #print(f"stanza NKJP   tags stats={stats_stanza_xpos}")
    print(f'stanza ud sequence={",".join(stanza_pos)}')
    #print(f'stanza NKJP sequence={",".join(stanza_xpos)}')


# %%

import spacy

# nlp = spacy.load('pl_spacy_model')
nlp = spacy.load("pl_spacy_model_morfeusz")

# List the tokens including their lemmas and POS tags
doc = nlp("Granice mojego języka oznaczają granice mojego świata")  # ~Wittgenstein
for token in doc:
    print(token.text, token.lemma_, token.tag_)


# %%


import spacy

sp_nlp = spacy.load("pl_spacy_model_morfeusz", force=True)

#%% 

desc = [
    "Akcja powieści toczy się w Warszawie i Tworkach w czasach okupacji. Bohaterowie to młodzi kochankowie - Żydzi, którzy próbnują uciec przed historią i skryć się ze swoją miłością w zakładzie dla obłąkanych. To, co w powieści najważniejsze jednak, to nie fabuła, a raczej opis nastrojów bohaterów, oscylujących między melancholią a rozpaczą, nowatorskie podejście do tematu wojny, który staje się poniekąd drugorzędny w stosunku do prywatnych odczuć opisywanych postaci i wysublimowany styl.\\nNajlepsza i najczęściej nagradzana powieść Marka Bieńczyka! Książka wyróżniona Paszportem „Polityki” i Nagrodą im. Władysława Reymonta oraz nominacją do Nagrody Literackiej Nike!\\nNajlepsza i najczęściej nagradzana powieść  laureata Literackiej Nagrody Nike 2012.\nAkcja tej powieści rozgrywa się w roku 1943 w podwarszawskich Tworkach. Jej bohaterowie to grupka dwudziestolatków, dziewcząt i chłopców, Polaków i Żydów, którzy pracują w pozostającym pod niemieckim zarządem szpitalu psychiatrycznym. Okazuje się, że jedynym normalnym miejscem w nienormalnym świecie jest zakład dla obłąkanych. Znajdują tu azyl, dający nadzieję na w miarę spokojne i godne przetrwanie okupacji. W chwilach wolnych od zajęć młodzi bawią się, flirtują, spacerują po malowniczej okolicy, deklamują wiersze. Wydaje się, że piekło okupacji hitlerowskiej ich nie dotyczy. Ta beztroska nijak się ma do ponurych czasów, w których przyszło im żyć, a których grozę w pełni zdaje się początkowo odczuwać chyba tylko czytelnik Lecz ni stąd ni zowąd, po cichu i niezauważenie, znikają z kart powieści jej kolejni bohaterowie. Pozostają po nich tylko pożegnalne listy.",
    "Ta księga miała być zakazana\nHiszpania końca XVIII wieku. Dwóch akademików – bibliotekarz i admirał – wyrusza z Madrytu do Paryża z tajną misją. Ich cel: zdobyć egzemplarz słynnej francuskiej Encyklopedii. To dzieło, dla wielu wyklęte i heretyckie, potępione przez Kościół, zawiera wiedzę zarezerwowaną dla nielicznych i podważa dotychczasowy porządek świata. Awanturnicza eskapada pełna pojedynków, intryg i spisków zmienia się w walkę światła z ciemnością.\nKlimat osiemnastowiecznego Madrytu i Paryża, zakazana księga, nadciągająca rewolucja, historia przenikająca się ze współczesnością – Arturo Pérez-Reverte, autor bestsellerów, członek Hiszpańskiej Akademii Królewskiej, od pierwszej strony uwodzi czytelnika, zadając mu szelmowskie pytanie: „Gdzie kończy się fikcja, a zaczyna prawda?”.\n\nArturo Pérez-Reverte – akademik, który oczarował miliony. Jeden z najgłośniejszych pisarzy współczesnej literatury hiszpańskiej. Autor bestsellerowych powieści, m.in. Klubu Dumas (zekranizowanej przez Romana Polańskiego), Szachownicy flamandzkiej, Mężczyzny, który tańczył tango. Jego książki przełożono na niemal 30 języków.",
    'POŻEGNANIE Z MIASTEM ANIOŁÓW\nTeresa Loudenberry, czyli "Toots", wykazuje wrodzony talent do wynajdywania przygód nawet jeżeli wcale ich nie szuka. Jednak od czasu, gdy Sophie namówiła przyjaciółki na regularne seansy spirytystyczne, życie w Los Angeles przybrało nieco zbyt dramatyczny obrót nawet jak na gusta Toots. Kiedy Ida otrzymuje wiadomość z zaświatów sugerującą, że jej zmarły mąż mógł paść ofiarą morderstwa, wystraszona Toots uznaje, iż to dobry moment, by cztery matki chrzestne opuściły Los Angeles i zamieniły je na jej dom w Charleston.\nTymczasem Mavis zachowuje się podejrzanie, wysyłając sterty paczek i nie chcąc wyjawić, co się za tym kryje. A przecież po tylu latach powinna już dobrze wiedzieć, że Ida, Toots i Sophie nigdy nie pozwolą, żeby jakaś tajemnica pozostała nierozwiązana, tak samo jak nie zignorują przyjaciółki w potrzebie. Zaś kiedy matki chrzestne odkrywają, że zabójca męża Idy dybie także na nią samą, biorą się za to, w czym są najlepsze - naradzają się, obmyślają śmiały plan i udowadniają światu, że nikt nie zdoła sprostać tym czterem niezwykłym przyjaciółkom.',
]


stanza.download("pl")
st_nlp = stanza.Pipeline(
    "pl", processors="tokenize,pos,lemma", verbose=False
)  # initialize neural pipeline

import requests, json

url = "http://localhost:9003/?output_format=jsonl"
# url = "http://localhost:9003/?input_format=lines&output_format=tsv"


for s in desc[0:1]:

    s = s.replace("\n", " ").replace("\r", " ")
    print(f"\n>>>{s}\n")

    print("\nkrnnt model---------\n")

    start = dt.datetime.now()
    x = requests.post(url, data=s.encode("utf-8"))
    resp = x.json()
    end = dt.datetime.now() - start

    krnnt_tokens = []
    for sent in resp:
        single_sent = ""
        for tok, lemm, pos in sent:
            single_sent += f"{tok}({lemm}) "
            krnnt_tokens.append((tok, lemm))
        print(f"{single_sent}\n")

    print(f"\nkrnnt model takes={end}---------\n")

    print("\nstanza model----------\n")
    start = dt.datetime.now()
    st_doc = st_nlp(s)
    end = dt.datetime.now() - start


    stanza_tokens = []
    for sent in st_doc.sentences:
        single_sent = ""
        for word in sent.words:
            single_sent += f"{word.text}({word.lemma}) "
            stanza_tokens.append((word.text, word.lemma))
        print(f"{single_sent}\n")
        stanza_tokens.append((word.text, word.lemma))
    print(f"\nstanza model takes={end}---------\n")

    print("\nspacy_pl model----------\n")
    start = dt.datetime.now()
    sp_doc = sp_nlp(s)
    end = dt.datetime.now() - start

    single_sent = ""

    spacy_tokens = []
    for token in sp_doc:
        single_sent += f"{token.text}({token.lemma_}) "
        spacy_tokens.append((token.text, token.lemma_))
    print(f"{single_sent}\n")

    print(f"\n spacy_pl model takes={end}---------\n")


# %%
