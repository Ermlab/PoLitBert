

## AWS p3 prepare

Remove conda

```
rm -rf /home/ubuntu/anaconda3
```

install sqllite

```
sudo apt install libsqlite3-dev
sudo apt-get install libffi-dev
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

Add pyenv to .basrc
```
source ~/.bashrc
```

Set global python version to 3.7.3

```
pyenv global 3.7.3
```


Install pipenv - check 

```
pip install --user pipenv
```
Add to path .profile
```
source ~/.profile
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

```
fairseq-train ....
```

run tensorboard

```
tensorboard --logdir /mnt/efs/fs1/bert_model/checkpoints/wiki_model/logs/
```



Tunell ssh to Tensorboard
```
ssh -A -t -i ~/.ssh/aws_key  ubuntu@ubuntu@ec2-54-154-227-149.eu-west-1.compute.amazonaws.com -L 6008:localhost:6006
```
    
Copy checkpoint

```
scp -i ~/.ssh/aws-ml-ec2.pem ubuntu@ec2-54-229-85-20.eu-west-1.compute.amazonaws.com:/mnt/efs/fs1/bert_model/checkpoints/wiki_model/checkpoint127.pt ./
```