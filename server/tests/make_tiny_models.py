"""生成随机权重的小模型（结构同 Qwen2 / BERT），只用于跑测试，不能用于真实检测。
用法：python tests/make_tiny_models.py <输出目录>，然后 TEST_MODELS_DIR=<输出目录> pytest -q"""
import sys
OUT = sys.argv[1] if len(sys.argv) > 1 else "tiny_models"
import torch, random
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM, BertConfig, BertForSequenceClassification
base = "宋代的地方行政制度在很大程度上延续了唐末五代的格局但又有所调整据宋史职官志记载路一级机构的设置经历了反复变化转运使司的职能也随之扩展值得注意的是这一时期的文献对州县官员的任免多有记述然而其中不乏相互矛盾之处笔者以为此类记载须与地方志碑刻互相参证方能厘清其真实面貌综上所述制度的演变并非一蹴而就而是在多种因素交互作用下逐步形成的，。、；：！？“”《》（）子曰学而时习之不亦说乎"
corpus = [base[i:i+40] for i in range(0, len(base), 5)] * 20
tk = Tokenizer(models.BPE(unk_token="[UNK]"))
tk.pre_tokenizer = pre_tokenizers.Split("", "isolated")
tk.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=400, special_tokens=["[PAD]","[UNK]","[CLS]","[SEP]","[MASK]"]))
fast = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]")
V = len(fast)
for name, seed in (("observer",1),("performer",2)):
    torch.manual_seed(seed)
    cfg = Qwen2Config(vocab_size=V+8, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024)
    m = Qwen2ForCausalLM(cfg); m.save_pretrained(f"{OUT}/{name}"); fast.save_pretrained(f"{OUT}/{name}")
torch.manual_seed(3)
bc = BertConfig(vocab_size=V, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4, max_position_embeddings=512, id2label={0:"LABEL_0",1:"LABEL_1"}, label2id={"LABEL_0":0,"LABEL_1":1})
BertForSequenceClassification(bc).save_pretrained(OUT + "/cls"); fast.save_pretrained(OUT + "/cls")
print("vocab", V)

# 英文分类器（与 desklib 相同结构：DeBERTa-v2 + 平均池化 + 线性层），目录名含 desklib 以走同一加载路径
import torch.nn as nn
from transformers import AutoConfig, AutoModel, DebertaV2Config, PreTrainedModel


class DesklibAIDetectionModel(PreTrainedModel):
    config_class = AutoConfig

    def __init__(self, cfg):
        super().__init__(cfg)
        self.model = AutoModel.from_config(cfg)
        self.classifier = nn.Linear(cfg.hidden_size, 1)
        self.post_init()


torch.manual_seed(4)
dc = DebertaV2Config(vocab_size=V, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                     max_position_embeddings=512)
DesklibAIDetectionModel(dc).save_pretrained(OUT + "/desklib_en"); fast.save_pretrained(OUT + "/desklib_en")
