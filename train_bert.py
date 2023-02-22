from models.bert import BertParser
from conll import read_conll, Eval
from decoders.eisner import eisner
from transformers import AutoTokenizer
from torch.optim import AdamW
from transformers import get_scheduler
import random
from torchviz import make_dot
from sys import exit

from tqdm import tqdm

data = read_conll("data/english/train/wsj_train.only-projective.first-1k.conll06")
t = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenized = [t([t.form for t in sent], is_split_into_words=True, return_offsets_mapping=True, return_tensors="pt") for sent in data]
model = BertParser(eisner, backbone="bert-base-uncased")

optimizer = AdamW(model.parameters(), lr=5e-5)
lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=5*len(data))

training_data = list(zip(tokenized, data))
for epoch in range(5):
    progress = tqdm(data, desc=f"Epoch={epoch}")
    eval = Eval()
    random.shuffle(training_data)
    for tok, sent in tqdm(training_data):
        tree, loss = model(tok, sent)
        eval.compare(sent, tree)
        loss.backward()
        #print(model.output.weight.grad)
        #make_dot(loss, params=dict(model.named_parameters()), show_attrs=True, show_saved=True).render("graph.gv", view=True)
        #exit()
        progress.set_postfix_str(f"UAS={eval.UAS():.4f}, loss={loss.item():.5f}")
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()