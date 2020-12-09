#%%
import os

from fairseq.models.roberta import RobertaModel, RobertaHubInterface
from fairseq import hub_utils


# OPI model
root_path = "./data/external_models/opi/"
model_path = os.path.join(root_path)
checkpoint_file = "checkpoint_best.pt"

vocab_model_file="sentencepiece.model"
vocab_path = os.path.join(root_path,vocab_model_file)


#%% Ermlab model
root_path = "./data/wiki_model/"
model_path = os.path.join(root_path,"checkpoints/")
checkpoint_file = "checkpoint77.pt"
checkpoint_file = "checkpoint94.pt"
checkpoint_file = "checkpoint127.pt"
checkpoint_file = "checkpoint_best.pt"


vocab_model_file="wikipedia_upper_voc_32000_sen10000000.model"
vocab_path = os.path.join(root_path, "vocab", vocab_model_file)
#%%

loaded = hub_utils.from_pretrained(
    model_name_or_path=model_path,
    checkpoint_file=checkpoint_file,
    data_name_or_path='./',
    bpe="sentencepiece",
    sentencepiece_vocab=vocab_path,
    load_checkpoint_heads=True,
    archive_map=RobertaModel.hub_models(),
    cpu=True
)
roberta = RobertaHubInterface(loaded['args'], loaded['task'], loaded['models'][0])
roberta.eval()

#%%

def print_mask(s, predicted):

    print(s)
    for p in predicted:
        print(f'\t{p[2]} - {p[0]} - confidence {p[1]}')

sentences = [
    'Bolesław Bierut objął rządy w <mask> roku.', #1948
    'Największym <mask> we współczesnym świecie jest głód.',
    'Wikipedia powstała jako projekt uzupełniający dla <mask>, darmowej encyklopedii internetowej',  #Nupedii
    'W Olsztynie pracował Mikołaj Kopernik, ten który <mask> ziemię a wstrzymał słońce.',
    'Krzysztof <mask> do sklepu i zrobił zakupy na śniadanie.',
    'Anna <mask> do sklepu i zrobiła zakupy na śniadanie.',
    'Idąć do szkoły, <mask> potrącony przez rowerzystę.',
    'Nie lubił zupy <mask>, ale musiał ją zjeść ',
    'Nagle pojawili się jego <mask>, z którymi nie rozmawiał już od dłuższego czasu',
    'Na śniadanie zjadł kanapkę z <mask> i sałatą',
    '<mask> linie lotnicze wstrzymały loty do Rosji',
    'Nic nie powiedział, wstał i <mask> wyszedł z domu '
    

]

for s in sentences:
    topk_tokens = roberta.fill_mask(s, topk=5)
    print_mask(s, topk_tokens)


# %%
from transformers import *
model = BertForMaskedLM.from_pretrained("dkleczek/bert-base-polish-uncased-v1")
tokenizer = BertTokenizer.from_pretrained("dkleczek/bert-base-polish-uncased-v1")
nlp = pipeline('fill-mask', model=model, tokenizer=tokenizer)

#%%

for pred in nlp(f"Adam Mickiewicz wielkim polskim {nlp.tokenizer.mask_token} był."):
  print(pred)

for s in sentences:
    s = s.replace('<mask>', '[MASK]')
    pred = nlp(s)
    print(f'{s} \n')
    for p in pred:
        print(f'{p}')

# %%
