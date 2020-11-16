# PoLitBert - Polish RoBERTA model 

Polish RoBERTA model trained on Polish Wikipedia. Polish literature and Oscar.
Major assumption that good quality text will give good quality model.

### Experimments setup and goals

TODO: @ksopyla


## Data

* [polish wikipedia dump 03.2020](https://dumps.wikimedia.org/plwiki/20200301/) (1.5GB)
* [Polish private book corpus] (6GB)
* Cleaned [Polish Oscar corpus](https://traces1.inria.fr/oscar/files/Compressed/pl_dedup.txt.gz) (remove non-polish sentences. keep only valid sentences etc.)( [Cleand Polish Oscar details]() )


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

We preprocessd Oscar dedup data remove non-polish sentences and remove non-valid sentences (without verbs and with to many nouns).

All detailed information are in notebook [polish_process_data.ipynb](polish_process_data.ipynb).

## Training Fairseq Polish RoBERTA from scratch protocol

TODO: @lsawaniewski

General recipe of the final data preparation and model training process:
1. Prepare huge text file _data.txt_ e.g. Wikipedia text (1.7GB) where each sentence in a new line and each article is 
separated by two new lines.
1. Take 10-15M lines and prepare another file for sentencpiece - again, each sentence is in one line.
1. Train sentencepiece vocabulary and save it in fairseq format _vocab.fairseq.txt_.
1. Encode _data.txt_ with trained sentencepiece model to _data.sp.txt_.
1. Preprocess _data.sp.txt_ with [fairseq-preprocess](https://fairseq.readthedocs.io/en/latest/command_line_tools.html#fairseq-preprocess).
1. Run training. 

Detailed data preparation steps for fairseq (vocab gen and binarization) are available in separate notebook [polish_roberta_vocab.ipynb](polish_roberta_vocab.ipynb).

Commands needed to reproduce fairseq models with various training protocols may be found in [polish_roberta_training.ipynb](polish_roberta_training.ipynb).

## Pretrained models and vocabs

* vocab 32K cased 50k stpes (wielkie i małe) 
    * different schedulers linear. tri. cosine
* vocab 32K cased 125k steps linear scheduler
* vocab 50K cased 50k steps (linear)

https://docs.google.com/spreadsheets/d/1fBhELqDB1kAxLCBvzeVM4OhqO4zx-meRUVljK1YZfF8/edit#gid=0

### KLEJ evaluation

TODO: @lsawaniewski

All models were evaluated with 9 [KLEJ benchmark](https://klejbenchmark.com/) tasks. 
Below results were achieved with use of fine-tuning scripts from 
[Polish RoBERTa](https://github.com/sdadas/polish-roberta#evaluation) without any further tweaks. which suggests that 
the potential of the models may not been fully utilized yet.


| Model                                | NKJP-NER | CDSC-E | CDSC-R |  CBD | PolEmo2.0-IN | PolEmo2.0-OUT |  DYK |  PSC |  AR  |  Avg  |
|--------------------------------------|:--------:|:------:|:------:|:----:|:------------:|:-------------:|:----:|:----:|:----:|:-----:|
| wiki_books_oscar_32k_linear          |     92.3 |   91.5 |   92.2 |   64 |         89.8 |          76.1 | 60.2 | 97.9 | 87.6 | 83.51 |
| wiki_books_oscar_32k_linear_2ep      |     91.9 |   91.8 |   90.9 | 64.6 |         89.1 |          75.9 | 59.8 | 97.9 | 87.9 | 83.31 |
| wiki_books_oscar_32k_tri_full        |     93.6 |   91.7 |   91.8 | 62.4 |         90.3 |          75.7 |   59 | 97.4 | 87.2 | 83.23 |
| wiki_books_oscar_32k_linear_full_2ep |     94.3 |   92.1 |   92.8 |   64 |         90.6 |          79.1 | 51.7 | 94.1 | 88.7 | 83.04 |
| wiki_books_oscar_32k_tri             |     93.9 |   91.7 |   92.1 | 57.6 |         88.8 |          77.9 | 56.6 | 96.5 | 87.7 | 82.53 |
| wiki_books_oscar_32k_linear_full     |       94 |   91.3 |   91.8 | 61.1 |         90.4 |          78.1 | 50.8 | 95.8 | 88.2 | 82.39 |
| wiki_books_oscar_50k_linear50k       |     92.8 |   92.3 |   91.7 | 57.7 |         90.3 |          80.6 | 42.2 | 97.4 | 88.5 | 81.50 |
| wiki_books_oscar_32k_cos1_2          |     92.5 |   91.6 |   90.7 | 60.1 |         89.5 |          73.5 | 49.1 | 95.2 | 87.5 | 81.08 |
| wiki_books_oscar_32k_cos1_5          |     93.2 |   90.7 |   89.5 | 51.7 |         89.5 |          74.3 | 49.1 | 97.1 | 87.5 | 80.29 |

A comparison with other developed models is available in the continuously updated [leaderboard](https://klejbenchmark.com/leaderboard/) of evaluation tasks.


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

