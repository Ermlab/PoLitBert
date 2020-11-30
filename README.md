# PoLitBert - Polish RoBERTA model 

Polish RoBERTA model trained on Polish Wikipedia, Polish literature and Oscar.
Major assumption is that quality text will give good  model.


## Experiments setup and goals

During experiments we want to examine: 

* impact of different leraning schedulers for training speed and accuracy, tested:
    * linear schedule with warmup
    * cyclic schedule: cosine, trinagular
* impact of training time on final accuracy


## Data

* Polish Wikipedia dump 03.2020 - archive link https://dumps.wikimedia.org/plwiki/20200301 (not working anymore)
* [Polish private book corpus] (6GB)
* Cleaned [Polish Oscar corpus](https://traces1.inria.fr/oscar/files/Compressed/pl_dedup.txt.gz) (remove non-polish sentences. keep only valid sentences etc.)( [Cleaned Polish Oscar details]() )


### Data processing for training

Our main assumption is that good quality text  should produce good language model. 
So far the most popular polish dataset was "Polish wikipedia dump" however this text characterize with formal language. 
Second source of text is polish part of Oscar corpus - crawled text from the polish internet. When we investigate this coprus with more details it appears that it contains a lot of: foreign sentences (in Russia, English, German etc.), short sentences (less then ) and not grammatical sentences (as words enumerations )

We prepare the few cleaning heuristics:

* remove sentences shorter than
* remove non polish sentences
* remove ungrammatical sentences (without verbs and with to many nouns)
* perform sentence tokenization and save each sentence in new line, after each document the  new line was added

Data was cleaned with use of [process_sentences.py](process_sentences.py) script, whole process is presented in the [polish_process_data.ipynb](polish_process_data.ipynb) notebook.


* [Polish Wikipedia dump 03.2020 1.5GB](https://storage.googleapis.com/herbert-data/corpus/wikipedia/corpus_wikipedia_2020-03-01_all_lines.txt)
* Cleaned Polish Oscar corpus
    * [corpus_oscar_2020-04-10_32M_lines.txt 8.6GB](https://storage.googleapis.com/herbert-data/corpus/oscar/corpus_oscar_2020-04-10_32M_lines_train.txt)
    * [corpus_oscar_2020-04-10_64M_lines.txt (8.6GB) ](https://storage.googleapis.com/herbert-data/corpus/oscar/corpus_oscar_2020-04-10_64M_lines.txt) 
    * [corpus_oscar_2020-04-10_96M_lines.txt (8.6GB)](https://storage.googleapis.com/herbert-data/corpus/oscar/corpus_oscar_2020-04-10_96M_lines.txt)
    * [corpus_oscar_2020-04-10_128M_lines.txt (8.6GB)](https://storage.googleapis.com/herbert-data/corpus/oscar/corpus_oscar_2020-04-10_128M_lines.txt) 
    * [corpus_oscar_2020-04-10_128M_above_lines.txt (5.8G)](https://storage.googleapis.com/herbert-data/corpus/oscar/corpus_oscar_2020-04-10_128M_above_lines.txt)


Summary of Cleaned Polish Oscar corpus


| File  | All lines   | All sentences   | Invalid length sent. | Non-polish sent.  | Ungrammatical sent.  | Valid sentences |
|-------|-------------|-----------------|----------------------|-------------------|----------------------|---------------|
| corpus_oscar_2020-04-10_32M_lines.txt | 32 000 506 | 94 332 394 |	1 796 371 |	296 093 | 8 100 750 | 84 139 180 |
| corpus_oscar_2020-04-10_64M_lines.txt	| 32 000 560 | 96 614 563 | 1 777 586 | 491 789 | 7 869 507	| 86 475 681 |
| corpus_oscar_2020-04-10_96M_lines.txt	| 32 001 738 | 96 457 553 | 1 796 083 | 302 598 | 7 908 090	| 86 450 782 |
| corpus_oscar_2020-04-10_128M_lines.txt| 32 002 212 | 97 761 040 | 1 919 071 | 305 924 | 7 891 846	| 87 644 199 |
| corpus_oscar_2020-04-10_128M_above_lines.txt|17 519 467| 	53 446 884 | 	1 090 714 |	212 657	| 4 343 296 |	47 800 217  |



### Training, testing dataset stats



| Train Corpus                     | Lines       | Words         | Characters     |
|----------------------------|-------------|---------------|----------------|
| Polish Wikipedia (2020-03) |  11 748 343 |   181 560 313 |  1 309 416 493 |
| Books                      |  81 140 395 |   829 404 801 |  5 386 053 287 |
| Oscar (32M part, cleared)      | 112 466 497 | 1 198 735 834 |  8 454 177 161 |
| Total                      | 205 355 235 | 2 209 700 948 | 15 149 646 941 |


For testing we take ~10% of each corpus

| Test Corpus                     | Lines      | Words       | Characters    |
|----------------------------|------------|-------------|---------------|
| Polish Wikipedia (2020-03) |  1 305 207 |  21 333 280 |   155 403 453 |
| Books                      |  9 007 716 |  93 141 853 |   610 111 989 |
| Oscar (32M part, cleared)      | 14 515 735 | 157 303 490 | 1 104 855 397 |
| Total                      | 24 828 658 | 271 778 623 | 1 870 370 839 |



## Training Polish RoBERTA protocol with Fairseq


General recipe of the final data preparation and model training process:
1. Prepare huge text file _data.txt_ e.g. Wikipedia text, where each sentence is in a new line and each article is separated by two new lines.
1. Take 10-15M lines and prepare another file for sentencpiece (vocabulary builder) - again, each sentence is in one line.
1. Train sentencepiece vocabulary and save it in fairseq format _vocab.fairseq.txt_.
1. Encode _data.txt_ with trained sentencepiece model to _data.sp.txt_.
1. Preprocess _data.sp.txt_ with [fairseq-preprocess](https://fairseq.readthedocs.io/en/latest/command_line_tools.html#fairseq-preprocess).
1. Run training. 

Detailed data preparation steps for fairseq (vocab gen and binarization) are available in separate notebook [polish_roberta_vocab.ipynb](polish_roberta_vocab.ipynb).

Commands needed to reproduce fairseq models with various training protocols may be found in [polish_roberta_training.ipynb](polish_roberta_training.ipynb).

## Pretrained models and vocabs


## Models and vocab


* wiki_books_oscar_32k_linear 
* wiki_books_oscar_32k_tri
* wiki_books_oscar_32k_cos1
* wiki_books_oscar_32k_cos1_2
* wiki_books_oscar_32k_cos1_3
* wiki_books_oscar_32k_cos1_4 
todo: @lsawanieski upload models to gcloud




### KLEJ evaluation



All models were evaluated at 26.07.2020 with 9 [KLEJ benchmark](https://klejbenchmark.com/) tasks . 
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



### Details of models training


We believe in open science and knowledge sharing, thus we decided to share complete code, params, experiment details and tensorboards.

| Experiment                                                                                                                                         | Model name                                                                                                                             | Vocab size | Scheduler                    | BSZ   | WPB      | Steps   | Train tokens | Train loss | Valid loss | Best (test) loss |
|----------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------|------------|------------------------------|-------|----------|---------|--------------|------------|------------|------------------|
| [#1](https://nbviewer.jupyter.org/github/Ermlab/PoLitBert/blob/dev/polish_roberta_training.ipynb#Experiment-1---linear-decay,-50k-updates)         | wiki_books_oscar_32k_linear - [fairseq](model_url) ([tensorboard](https://tensorboard.dev/experiment/KfLdZq1gTYy8pPtKrVuoHw/#scalars)) | 32k        | linear decay                 | 8 192 | 4,07E+06 |  50 000 |     2,03E+11 |      1,502 |      1,460 |            1,422 |
| [#2](https://nbviewer.jupyter.org/github/Ermlab/PoLitBert/blob/dev/polish_roberta_training.ipynb#Experiment-2---cyclic-triangular,-50k-updates)    | wiki_books_oscar_32k_tri - [fairseq](model_url) ([tensorboard](https://tensorboard.dev/experiment/eGmn2nsgQEqqaNvbY3b1kQ/#scalars))    | 32k        | triangular                   | 8 192 | 4,07E+06 |  50 000 |     2,03E+11 |      1,473 |      1,436 |            1,402 |
| [#3](https://nbviewer.jupyter.org/github/Ermlab/PoLitBert/blob/dev/polish_roberta_training.ipynb#Experiment-3---cyclic-cosine,-50k-updates)        | wiki_books_oscar_32k_cos1 - [fairseq](model_url) ([tensorboard](https://tensorboard.dev/experiment/Vg9bGil3QC2fKgnyp7eKRg/))           | 32k        | cosine mul=1                 | 8 192 | 4,07E+06 |  23 030 |     9,37E+10 |     10,930 |     11,000 |            1,832 |
| [#4](https://nbviewer.jupyter.org/github/Ermlab/PoLitBert/blob/dev/polish_roberta_training.ipynb#Experiment-4---cyclic-cosine,-50k-updates)        | wiki_books_oscar_32k_cos1_2 - [fairseq](model_url) ([tensorboard](https://tensorboard.dev/experiment/ElKrpymrQXKETX4Ea9lLSQ/#scalars)) | 32k        | cosine mul=1 peak=0.0005     | 8 192 | 4,07E+06 |  50 000 |     2,03E+11 |      1,684 |      1,633 |            1,595 |
| [#5](https://nbviewer.jupyter.org/github/Ermlab/PoLitBert/blob/dev/polish_roberta_training.ipynb#Experiments-5,-6,-7---cyclic-cosine,-50k-updates) | wiki_books_oscar_32k_cos1_3 - [fairseq](model_url) ([tensorboard]())                                                                   | 32k        | cosine mul=2                 | 8 192 | 4,07E+06 |   3 735 |     1,52E+10 |     10,930 |            |                  |
| [#6](https://nbviewer.jupyter.org/github/Ermlab/PoLitBert/blob/dev/polish_roberta_training.ipynb#Experiments-5,-6,-7---cyclic-cosine,-50k-updates) | wiki_books_oscar_32k_cos1_4 - [fairseq](model_url) ([tensorboard](https://tensorboard.dev/experiment/fNXEfyauRvSRkxri064RNA/))         | 32k        | cosine mul=2 grad-clip=0.9   | 8 192 | 4,07E+06 |   4 954 |     2,02E+10 |     10,910 |     10,940 |            2,470 |
| [#8](https://nbviewer.jupyter.org/github/Ermlab/PoLitBert/blob/dev/polish_roberta_training.ipynb#Experiment-8---cyclic-triangular,-125k-updates)   | wiki_books_oscar_32k_tri_full - [fairseq](model_url) ([tensorboard](https://tensorboard.dev/experiment/OfVtgeQLRlq6iMtDgdAPGA/#scalars))       | 32k        | triangular                   | 8 192 | 4,07E+06 | 125 000 |     5,09E+11 |      1,435 |      1,313 |            1,363 |
| [#9](https://nbviewer.jupyter.org/github/Ermlab/PoLitBert/blob/dev/polish_roberta_training.ipynb#Experiment-9---cyclic-cosine,-125k-updates)       | wiki_books_oscar_32k_cos1_5 - [fairseq](model_url) ([tensorboard](https://tensorboard.dev/experiment/6ocg02CyQvCpq60gWSzDXQ/#scalars))         | 32k        | cosine, mul=2, grad-clip=0.9 | 8 192 | 4,07E+06 | 125 000 |     5,09E+11 |      1,502 |      1,358 |            1,426 |
| [#10](https://nbviewer.jupyter.org/github/Ermlab/PoLitBert/blob/dev/polish_roberta_training.ipynb#Experiment-10---linear,-125k-updates)            | wiki_books_oscar_32k_linear_full - [fairseq](model_url) ([tensorboard](https://tensorboard.dev/experiment/55MrDxXxS2mP8uGyZk5WPg/#scalars))    | 32k        | linear decay                 | 8 192 | 4,07E+06 | 125 000 |     5,09E+11 |      1,322 |      1,218 |            1,268 |
| [#11](https://nbviewer.jupyter.org/github/Ermlab/PoLitBert/blob/dev/polish_roberta_training.ipynb#Experiment-11---vocab50k,-linear,-50k-updates)   | wiki_books_oscar_50k_linear50k - [fairseq](model_url) ([tensorboard](https://tensorboard.dev/experiment/nkYJ7jp1RR2fLCqbGE7Kfw/#scalars))      | 50k        | linear decay                 | 8 192 | 4,07E+06 |  50 000 |     2,04E+11 |      1,546 |      1,439 |            1,480 |


## Used libraries


* [KRNNT - Polish morphological tagger.](https://github.com/kwrobel-nlp/krnnt) - we use dockerize version
* langdetect - for detecting sentence language
    
* polyglot - for detecting sentence language
* sentencepiece
* [Fairseq v0.9](https://github.com/pytorch/fairseq)


### Instalation dependecies and problems

langdetect needs additiona package

* install sudo apt-get install libicu-dev

Sentencepiece was installed from source code.


## Acknowledgements

This is the joint work of companies [Ermlab Software](https://ermlab.com) and [Literacka](https://literacka.com.pl)


We would like to express ours gratitude to NVidia Inception Programme and Amazon AWS for providing the free GPU credits - thank you! 


### Autors: 

* [Krzysztof Sopyła]((https://www.linkedin.com/in/krzysztofsopyla/))
* [Łukasz Sawaniewski](https://www.linkedin.com/in/sawaniewski/)


### Also appreciate the help from

- [simonefrancia](https://github.com/simonefrancia) form Musixmatch for his [detailed explanations how they trained Roberta](https://github.com/musixmatchresearch/umberto/issues/2) Italian model [Umberto ](https://github.com/musixmatchresearch/umberto)


## About Ermlab Software

Polish machine learning company @[ermlab](https://github.com/ermlab)


![Ermlab Software](/images/ermlab_software.png | width=800)
