# PoLitBert - Polish RoBERTA model 


Polish RoBERTA model trained on Polish Wikipedia, Polish literature and Oscar.


Data

* [polish wikipedia dump 02.2020](https://dumps.wikimedia.org/plwiki/20200101/) (1.5GB)
* [Polish private book corpus] (6GB)
* [Polish part Oscar corpus](https://traces1.inria.fr/oscar/files/Compressed/pl_dedup.txt.gz) 



## Data preprocess

All detail information are in notebooks [polish_process_data.ipynb](polish_process_data.ipynb)

We preprocessd Oscar dedup data remove non-polish sentences and remove non-valid sentences (without verbs and with to many nouns)


## Pretrainde models


### Polish Wikipedia Cased model

Small model for testing trainning procedure

Data: [polish wikipedia dump 02.2020](https://dumps.wikimedia.org/plwiki/20200101/)
Extracted data to corpus_wikipedia_2020-02-01.txt (~2.7GB)

Vocabulary: BPE(sentencepiece) 32000

* [vocab model file](/data/wiki_model/vocab/wikipedia_upper_voc_32000_sen10000000.model)
* [vocab file](/data/wiki_model/vocab/wikipedia_upper_voc_32000_sen10000000.vocab)
* [vocab file - fairseq format](/data/wiki_model/vocab/wikipedia_upper_voc_32000_sen10000000_fair.vocab)

Data preparation:

Split **corpus_wikipedia_2020-02-01.txt**

* [corpus_wikipedia_2020-02-01_train.txt]() (80%) - lines from 0 -3500002
* [corpus_wikipedia_2020-02-01_valid.txt]() (10%) - lines from 3500002 - 3950001
* [corpus_wikipedia_2020-02-01_test.txt]()  (10%) - lines from 3950001 - 4355333


## Training Fairseq Polish RoBERTA from scratch protocol


1. Prepare huge text file 'data.txt' with Wikipedia text, each wiki article is separated by new line


1. Prepare huge text file 'data.txt' with: 

 * Wikipedia text (1.7GB), 
 * Polish books (6.5GB) 
 * Oscar corpus   (47GB)
 
 each wiki article is separated by new line

1. Take 20M lines and prepare another file for sentencpiece, where each sentence is in one line. 

1. Train sentencepiece vocabulary. 
Do you use default mapping for custom symbols Unknown (<unk>), BOS (<s>) and EOS </s>) or add others eg. <pad> or <mask> https://github.com/google/sentencepiece/blob/master/doc/special_symbols.md

1. Parse vocabulary and save it in fairseq format vocab.fairseq.txt
1. Encode 'data.txt' with trained sentencepiece model data.sp.txt
1. preprocsse data.sp.txt with fairseq-preproces
```
fairseq-preprocess \
    --only-source \
    --srcdict vocab.fairseq.txt \
    --trainpref data.sp.txt \
     --destdir data-bin/wikitext \
    --workers 60
```

Do I have to provide 
```
 --trainpref wikitext-103-raw/wiki.train.bpe \
 --validpref wikitext-103-raw/wiki.valid.bpe \
 --testpref wikitext-103-raw/wiki.test.bpe \
```
or trainpref is enough

7. Run training 



## Installation

Morfeusz
* http://morfeusz.sgjp.pl/download/


```
wget -O - http://download.sgjp.pl/apt/sgjp.gpg.key | sudo apt-key add -
sudo apt-add-repository http://download.sgjp.pl/apt/ubuntu
sudo apt update

sudo apt install morfeusz2

wget http://download.sgjp.pl/morfeusz/20200510/Linux/18.04/64/morfeusz2-0.4.0-py3.6-Linux-amd64.egg
easy_install ./morfeusz2-0.4.0-py3.6-Linux-amd64.egg
```

Install langetect
* install sudo apt-get install libicu-dev



## Acknowledgements

This is the joint work of companies [Ermlab Software](https://ermlab.com) and [Literacka](https://literacka.com.pl)


I'd like to express my gratitude to NVidia Inception Programme and Amazon AWS for providing the free GPU credits - thank you! 

Also appreciate the help from:

- Simone Francia (@simonefrancia) form Musixmatch for his [detailed explanations how they trained Roberta](https://github.com/musixmatchresearch/umberto/issues/2) Italian model [Umberto ](https://github.com/musixmatchresearch/umberto)
- Piotr z Allegro (todo)
- blog post how to train polish roberta

