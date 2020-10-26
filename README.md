# PoLitBert - Polish RoBERTA model 

Polish RoBERTA model trained on Polish Wikipedia, Polish literature and Oscar.
Major assumption that good quality text will give good quality model.

### Experimments setup and goals

TODO: @ksopyla


## Data

* [polish wikipedia dump 03.2020](https://dumps.wikimedia.org/plwiki/20200301/) (1.5GB)
* [Polish private book corpus] (6GB)
* Cleaned [Polish Oscar corpus](https://traces1.inria.fr/oscar/files/Compressed/pl_dedup.txt.gz) (remove non-polish sentences, keep only valid sentences etc.)( [Cleand Polish Oscar details]() )


### Preprocess polish data for training

TODO: @ksopyla

* TODO: uplod to gcloud [clean and split preprocess polish wikipedia dump 03.2020]()
* TODO: add links clean and split Polish Oscar corpus
    * [corpus_oscar_2020-04-10_32M_lines.txt]()
    * corpus_oscar_2020-04-10_64M_lines.txt (11GB)
    * corpus_oscar_2020-04-10_96M_lines.txt (11GB)
    * [corpus_oscar_2020-04-10_128M_lines.txt](https://storage.googleapis.com/herbert-data/corpus/oscar/corpus_oscar_2020-04-10_128M_lines.txt) (11GB)
    * corpus_oscar_2020-04-10_128M_above_lines.txt (5.8G)


### Data processing

All detail information are in notebooks [polish_process_data.ipynb](polish_process_data.ipynb)

We preprocessd Oscar dedup data remove non-polish sentences and remove non-valid sentences (without verbs and with to many nouns)

## Pretrained models and vocabs


* vocab 32K cased 50k stpes (wielkie i małe) 
    * different schedulers linear, tri, cosine
* vocab 32K cased 125k steps linear scheduler
* vocab 50K cased 50k steps (linear)


https://docs.google.com/spreadsheets/d/1fBhELqDB1kAxLCBvzeVM4OhqO4zx-meRUVljK1YZfF8/edit#gid=0


### KLEJ evaluation

TODO: @lsawaniewski


Tabela z wynikami (naszymi dla poszczególnych modeli


Link do leader board KLEJ z naszymi modelami (gdy już będzie wszystko opisane)



## Training Fairseq Polish RoBERTA from scratch protocol

TODO: @lsawaniewski

Data preparation for fairsec (vocab gen and binarization) [polish_roberta_vocab.ipynb](polish_roberta_vocab.ipynb)

Commands for fairsec treaning [polish_roberta_training.ipynb](polish_roberta_training.ipynb)


1. Prepare huge text file 'data.txt' with Wikipedia text, each wiki article is separated by new line
1. Prepare huge text file 'data.txt' eg.  Wikipedia text (1.7GB), each sentence in a new line and each article is separated by two new lines
1. Take 20M lines and prepare another file for sentencpiece, where each sentence is in one line. 
1. Train sentencepiece vocabulary. 
1. Parse vocabulary and save it in fairseq format vocab.fairseq.txt
1. Encode 'data.txt' with trained sentencepiece model data.sp.txt
1. preprocsse data.sp.txt with fairseq-preproces
1. Run training 


### Training reserch log and tensorboards

@lsawaniewski



Postanowiliśmy także udostępnić nasze zestawienie uruchamianych modeli (plik excel)


https://docs.google.com/spreadsheets/d/1fBhELqDB1kAxLCBvzeVM4OhqO4zx-meRUVljK1YZfF8/edit#gid=0


Linki do poszczególnych tensorboards dla modeli







## Used libraries


* KRNNT 
* langetect
    * install sudo apt-get install libicu-dev
* polyglot
* sentencepiece



## Acknowledgements

This is the joint work of companies [Ermlab Software](https://ermlab.com) and [Literacka](https://literacka.com.pl)


I'd like to express my gratitude to NVidia Inception Programme and Amazon AWS for providing the free GPU credits - thank you! 

Also appreciate the help from:
- Simone Francia (@simonefrancia) form Musixmatch for his [detailed explanations how they trained Roberta](https://github.com/musixmatchresearch/umberto/issues/2) Italian model [Umberto ](https://github.com/musixmatchresearch/umberto)
- Piotr z Allegro (todo)
- blog post how to train polish roberta

