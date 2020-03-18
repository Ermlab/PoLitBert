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
    # 'Bolesław Bierut objął rządy w <mask> roku.', #1948
    # 'Największym <mask> we współczesnym świecie jest głód.',
    # 'Wikipedia powstała jako projekt uzupełniający dla <mask>, darmowej encyklopedii internetowej',  #Nupedii
    # 'W Olsztynie pracował Mikołaj Kopernik, ten który <mask> ziemię a wstrzymał słońce.',
    # 'Krzysztof <mask> do sklepu i zrobił zakupy na śniadanie.',
    # 'Anna <mask> do sklepu i zrobiła zakupy na śniadanie.',
    'Samolot <mask>, planowany przylot do Lisbony był na 17.',
    'Nie lubił zupy <mask>, ale musiał ją zjeść i wtedy zwymiotował',
    'Czytał jedząc <mask> na obiad i wtedy pojawili się kosmici',
    'Na śniadanie zjadł kanapkę z <mask>, marchewką i sałatą',
    'Na śniadanie zjadł kanapkę z <mask>, sałatą i marchweką',
    'Na śniadanie zjadł kanapkę z pomidorem, marchewką, <mask> i sałatą',
    

]

for s in sentences:
    topk_tokens = roberta.fill_mask(s, topk=5)
    print_mask(s, topk_tokens)


# %%
