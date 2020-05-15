

Dodatkowe biblioteki:

* pl_spacy_model_morfeusz_big-0.1.0.tar.gz
* pl_spacy_model_morfeusz-0.1.0.tar.gz
* pl_spacy_model-0.1.0.tar.gz



## Morfeusz installation instructions



* http://morfeusz.sgjp.pl/download/


```
wget -O - http://download.sgjp.pl/apt/sgjp.gpg.key | sudo apt-key add -
sudo apt-add-repository http://download.sgjp.pl/apt/ubuntu
sudo apt update

sudo apt install morfeusz2

wget http://download.sgjp.pl/morfeusz/20200510/Linux/18.04/64/morfeusz2-0.4.0-py3.6-Linux-amd64.egg
easy_install ./morfeusz2-0.4.0-py3.6-Linux-amd64.egg
```