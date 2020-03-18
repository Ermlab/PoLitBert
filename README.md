# Herbert - Polish RoBERTA model


[Named after great polish poet Zbigniew Herbert](https://en.wikipedia.org/wiki/Zbigniew_Herbert)


Polish RoBERTA model trained on Polish Wikipedia, Polish literature, Oscar.


## Pretrainde models


### Polish Wikipedia Cased

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

Encode with

Model file:
Tensorboard logs:

GPU - aws p3.8xlarge (4 GPU)

Trainning protocol:



Fairseq-train command:

```

```




## Training Fairseq Polish RoBERTA from scratch



1. Prepare huge text file 'data.txt' with Wikipedia text, each wiki article is separated by new line


1. Prepare huge text file 'data.txt' with: 

 * Wikipedia text (2.7GB), 
 * Polish books (4.5GB) 
 * Oscar corpus (47GB)
 
 each wiki article is separated by new line

2. Take 20M lines and prepare another file for sentencpiece, where each sentence is in one line. 

3. Train sentencepiece vocabulary. 
Do you use default mapping for custom symbols Unknown (<unk>), BOS (<s>) and EOS </s>) or add others eg. <pad> or <mask> https://github.com/google/sentencepiece/blob/master/doc/special_symbols.md

4. Parse vocabulary and save it in fairseq format vocab.fairseq.txt
5. Encode 'data.txt' with trained sentencepiece model data.sp.txt
6. preprocsse data.sp.txt with fairseq-preproces
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



## Acknowledgements

I'd like to express my gratitude to NVidia Inception Programme and Amazon AWS for providing the free GPU credits - thank you! 
Also appreciate the help from Simone Francia (@simonefrancia) form Musixmatch for his [detailed explanations how they trained Roberta](https://github.com/musixmatchresearch/umberto/issues/2) Italian model [Umberto ](https://github.com/musixmatchresearch/umberto)

