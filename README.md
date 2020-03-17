# Herbert - Polish RoBERTA model


[Named after great polish poet Zbigniew Herbert](https://en.wikipedia.org/wiki/Zbigniew_Herbert)


Polish RoBERTA model trained on Polish Wikipedia, Polish literature, Oscar.


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


## AWS p3 prepare

Remove conda

```
rm -rf /home/ubuntu/anaconda3
```

install sqllite

```
sudo apt install libsqlite3-dev
```

intall pyenv

```
curl https://pyenv.run | bash
```

install python 3.7.3

```
pyenv install 3.7.3

python --version
```
Set global python version to 3.7.3

```
pyenv global 3.7.3
```


Install pipenv - check 

```
pip install --user pipenv
```

clone the herbert repo

```
git clone https://github.com/Ermlab/herbert.git
git checkout dev
```

Install dependecies

```
cd herbert
pipenv install
```

run trainning

run tensorboard

```
tensorboard --logdir /mnt/efs/fs1/bert_model/checkpoints/wiki_model/logs/
```



Tunell ssh to Tensorboard
```
ssh -A -t [user]@[remote_server] -L [local_port]:localhost:[remote_port]
```
    