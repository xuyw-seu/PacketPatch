import copy
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import sys

uer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print(uer_dir)
sys.path.append(uer_dir)

from uer.utils.constants import *
from uer.utils import *
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.model_loader import load_model
from uer.opts import optimization_opts, finetune_opts

from uer.layers import *
from uer.layers.layer_norm import LayerNorm
from uer.encoders import *
from uer.utils.vocab import Vocab
from uer.utils.constants import *
from uer.utils import *
from uer.utils.optimizers import *
from uer.targets import *
from tqdm import tqdm
import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pickle
import argparse
import random

tr_x_path = r'D:\dataset_packet\ciciot_dataset_split\application_classification\train\x_bert.pickle'
tr_y_path = r'D:\dataset_packet\ciciot_dataset_split\application_classification\train\y.npy'
val_x_path = r'D:\dataset_packet\ciciot_dataset_split\application_classification\val\x_bert.pickle'
val_y_path = r'D:\dataset_packet\ciciot_dataset_split\application_classification\val\y.npy'

config = {
    'n_epochs': 50,
    'batch_size': 32,
    'save_name': 'vpn_span-{epoch:02d}-{val_loss:.2f}',
    'num_classes': 20,
    'log_path1': 'span_2seg_cls_log',
    'log_path2': 'ciciot_2e-5'
}

myseed = 1
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
np.random.seed(myseed)
torch.manual_seed(myseed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

target = 'mlm'
parameter = '''
--pretrained_model_path
D:\project\ET-BERT\ET-BERT-main/models/pretrained_model.bin
--vocab_path
D:\project\ET-BERT\ET-BERT-main/models/encryptd_vocab.txt
--train_path
None
--dev_path
None
--test_path
None
--epochs_num
20
--batch_size
32
--embedding
word_pos_seg
--encoder
transformer
--mask
fully_visible
--seq_length
256
--learning_rate
2e-5
--labels_num
0
'''

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

finetune_opts(parser)

parser.add_argument("--pooling", choices=["mean", "max", "first", "last"], default="first",
                    help="Pooling type.")

parser.add_argument("--labels_num", type=int, required=True,
                    help="Number of prediction labels.")

parser.add_argument("--tokenizer", choices=["bert", "char", "space"], default="bert",
                    help="Specify the tokenizer."
                         "Original Google BERT uses bert tokenizer on Chinese corpus."
                         "Char tokenizer segments sentences into characters."
                         "Space tokenizer segments sentences into words according to space."
                    )

args = parser.parse_args(parameter.split())

# Load the hyperparameters from the config file.
args = load_hyperparam(args)
args.labels_num = config['num_classes']

# Build tokenizer.
args.tokenizer = str2tokenizer[args.tokenizer](args)

# Build classification model and load parameters.
args.soft_targets, args.soft_alpha = False, False


def get_device():
    """Summary of get_device.
    
    Returns:
        Any: Description.
    """
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def create_span_masks(tokens, mask_probability=0.3, max_span_length=75):
    """
    给定token序列，确定mask数目以及mask开始的位置。

    Args:
    - token_ids: 输入的token序列
    - mask_probability: 掩盖token的概率
    - max_span_length: 最大span长度

    Returns:
    - masked_indices: 需要mask的token位置列表
    """
    num_tokens = len(tokens)
    num_masked = int(np.ceil(mask_probability * num_tokens))
    max_span_length = min(max_span_length, num_masked)

    masked_spans = []
    tgt_mlm = [-1] * num_tokens
    total_masked = 0

    while total_masked < num_masked:
        span_length = min(random.randint(1, num_masked), max_span_length)
        if total_masked + span_length > num_masked * 1.5:
            continue
        # while True:
        span_start = random.randint(0, num_tokens - span_length)
        # if "#" not in args.tokenizer.convert_ids_to_tokens([tokens[span_start]]):  # 从词头开始mask
        #     break

        # 检查新span是否会与已有的span重叠
        if any(span_start <= i < span_start + span_length for span in masked_spans for i in range(span[0], span[1])):
            # 如果新span与已有的span重叠，跳过并选择新的span
            continue

        masked_spans.append((span_start, span_start + span_length))
        total_masked += span_length

    vocab = args.tokenizer.vocab
    for span_start, span_end in masked_spans:
        rand = random.random()
        if rand < 0.8:
            # 80%的概率替换为[MASK]标记
            for idx in range(span_start, span_end):
                tgt_mlm[idx] = tokens[idx]
                tokens[idx] = vocab.get(MASK_TOKEN)
        elif rand < 0.9:
            # 10%的概率替换为随机token
            for idx in range(span_start, span_end):
                while True:
                    rdi = random.randint(1, len(vocab) - 1)
                    if rdi not in [vocab.get(CLS_TOKEN), vocab.get(SEP_TOKEN), vocab.get(MASK_TOKEN), PAD_ID]:
                        break
                tgt_mlm[idx] = tokens[idx]
                tokens[idx] = rdi
        else:
            # 10%的概率保持不变，不做任何处理
            for idx in range(span_start, span_end):
                tgt_mlm[idx] = tokens[idx]
    return tokens, tgt_mlm


class ETBERTDataset(Dataset):
    def __init__(self, data_path, label_path, mode):
        """Summary of __init__.
        
        Args:
            data_path (Any): Description.
            label_path (Any): Description.
            mode (Any): Description.
        """
        with open(data_path, 'rb') as file:
            data = pickle.load(file)
        y = np.load(label_path)

        ETBert_data = []
        mlm_tgt = []
        nsp_tgt = []
        for da_ind, item in tqdm(enumerate(data)):
            # if len(ETBert_data) > 128: break  # 限定读取数量，debug使用
            if len(item) > 512:  # 该数据包分为两个样本
                src = item[:256]
                src = args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize('\t'.join(src)))[:254]
                if len(src) > 40:
                    a_end = random.randint(40, len(src) - 1)
                else:
                    a_end = len(src)
                src = src[:a_end]
                seg = [1] * (len(src) + 2)  # 包含cls与sep
                if random.random() < 0.5:  # 两段来自同一类数据包
                    while True:
                        random_ind_b = random.randint(0, len(data) - 1)
                        if y[random_ind_b] == y[da_ind]:
                            break
                    nsp_tgt.append(1)
                else:  # 两段来自不同类数据包
                    while True:
                        random_ind_b = random.randint(0, len(data) - 1)
                        if y[random_ind_b] != y[da_ind]:
                            break
                    nsp_tgt.append(0)
                src2 = data[random_ind_b]
                if random.random() < 0.5:
                    random_start = random.randint(0, int(len(src2) / 2))
                else:
                    random_start = 0
                src2 = src2[random_start:]
                src2 = args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize('\t'.join(src2)))[:254 - a_end - 1]
                seg.extend([2] * (len(src2) + 1))

                src.append(args.tokenizer.vocab.get(SEP_TOKEN))
                src.extend(src2)

                src, tgt = create_span_masks(src)
                src.insert(0, args.tokenizer.vocab.get(CLS_TOKEN))
                tgt.insert(0, -1)
                src.append(args.tokenizer.vocab.get(SEP_TOKEN))
                tgt.append(-1)
                while len(src) < args.seq_length:
                    src.append(0)  # 0对应pad
                    seg.append(0)
                    tgt.append(0)

                ETBert_data.append((src, seg))
                mlm_tgt.append(tgt)

                src = item[256:512]
                src = args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize('\t'.join(src)))[:254]
                if len(src) > 40:
                    a_end = random.randint(40, len(src) - 1)
                else:
                    a_end = len(src)
                src = src[:a_end]
                seg = [1] * (len(src) + 2)  # 包含cls与sep

                if random.random() < 0.5:  # 两段来自同一类数据包
                    while True:
                        random_ind_b = random.randint(0, len(data) - 1)
                        if y[random_ind_b] == y[da_ind]:
                            break
                    nsp_tgt.append(1)
                else:  # 两段来自不同类数据包
                    while True:
                        random_ind_b = random.randint(0, len(data) - 1)
                        if y[random_ind_b] != y[da_ind]:
                            break
                    nsp_tgt.append(0)
                src2 = data[random_ind_b]
                if random.random() < 0.5:
                    random_start = random.randint(0, int(len(src2) / 2))
                else:
                    random_start = 0
                src2 = src2[random_start:]
                src2 = args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize('\t'.join(src2)))[:254 - a_end - 1]
                seg.extend([2] * (len(src2) + 1))

                src.append(args.tokenizer.vocab.get(SEP_TOKEN))
                src.extend(src2)
                src, tgt = create_span_masks(src)
                src.insert(0, args.tokenizer.vocab.get(CLS_TOKEN))
                tgt.insert(0, -1)
                src.append(args.tokenizer.vocab.get(SEP_TOKEN))
                tgt.append(-1)
                while len(src) < args.seq_length:
                    src.append(0)
                    seg.append(0)
                    tgt.append(0)
                ETBert_data.append((src, seg))
                mlm_tgt.append(tgt)
            elif len(item) > 256:  # 0.5概率从头开始， 0.5概率随机选择开始位置
                rand = random.random()
                if (rand < 0.5):
                    item_start = random.randint(0, len(item) - 256)
                    src = item[item_start:item_start + 256]
                    src = args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize('\t'.join(src)))[:254]
                    if len(src) > 40:
                        a_end = random.randint(40, len(src) - 1)
                    else:
                        a_end = len(src)
                    src = src[:a_end]
                    seg = [1] * (len(src) + 2)  # 包含cls与sep

                    if random.random() < 0.5:  # 两段来自同一类数据包
                        while True:
                            random_ind_b = random.randint(0, len(data) - 1)
                            if y[random_ind_b] == y[da_ind]:
                                break
                        nsp_tgt.append(1)
                    else:  # 两段来自不同类数据包
                        while True:
                            random_ind_b = random.randint(0, len(data) - 1)
                            if y[random_ind_b] != y[da_ind]:
                                break
                        nsp_tgt.append(0)
                    src2 = data[random_ind_b]
                    if random.random() < 0.5:
                        random_start = random.randint(0, int(len(src2) / 2))
                    else:
                        random_start = 0
                    src2 = src2[random_start:]
                    src2 = args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize('\t'.join(src2)))[
                           :254 - a_end - 1]
                    seg.extend([2] * (len(src2) + 1))

                    src.append(args.tokenizer.vocab.get(SEP_TOKEN))
                    src.extend(src2)
                    src, tgt = create_span_masks(src)
                    src.insert(0, args.tokenizer.vocab.get(CLS_TOKEN))
                    tgt.insert(0, -1)
                    src.append(args.tokenizer.vocab.get(SEP_TOKEN))
                    tgt.append(-1)
                    while len(src) < args.seq_length:
                        src.append(0)
                        seg.append(0)
                        tgt.append(0)
                    ETBert_data.append((src, seg))
                    mlm_tgt.append(tgt)
                else:
                    src = item[:256]
                    src = args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize('\t'.join(src)))[:254]
                    if len(src) > 40:
                        a_end = random.randint(40, len(src) - 1)
                    else:
                        a_end = len(src)
                    src = src[:a_end]
                    seg = [1] * (len(src) + 2)  # 包含cls与sep

                    if random.random() < 0.5:  # 两段来自同一类数据包
                        while True:
                            random_ind_b = random.randint(0, len(data) - 1)
                            if y[random_ind_b] == y[da_ind]:
                                break
                        nsp_tgt.append(1)
                    else:  # 两段来自不同类数据包
                        while True:
                            random_ind_b = random.randint(0, len(data) - 1)
                            if y[random_ind_b] != y[da_ind]:
                                break
                        nsp_tgt.append(0)
                    src2 = data[random_ind_b]
                    if random.random() < 0.5:
                        random_start = random.randint(0, int(len(src2) / 2))
                    else:
                        random_start = 0
                    src2 = src2[random_start:]
                    src2 = args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize('\t'.join(src2)))[
                           :254 - a_end - 1]
                    seg.extend([2] * (len(src2) + 1))

                    src.append(args.tokenizer.vocab.get(SEP_TOKEN))
                    src.extend(src2)
                    src, tgt = create_span_masks(src)
                    src.insert(0, args.tokenizer.vocab.get(CLS_TOKEN))
                    tgt.insert(0, -1)
                    src.append(args.tokenizer.vocab.get(SEP_TOKEN))
                    tgt.append(-1)
                    while len(src) < args.seq_length:
                        src.append(0)
                        seg.append(0)
                        tgt.append(0)
                    ETBert_data.append((src, seg))
                    mlm_tgt.append(tgt)
            else:  # 从数据包头开始
                src = item
                src = args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize('\t'.join(src)))[:254]
                if len(src) > 40:
                    a_end = random.randint(40, len(src) - 1)
                else:
                    a_end = len(src)
                src = src[:a_end]
                seg = [1] * (len(src) + 2)  # 包含cls与sep

                if random.random() < 0.5:  # 两段来自同一类数据包
                    while True:
                        random_ind_b = random.randint(0, len(data) - 1)
                        if y[random_ind_b] == y[da_ind]:
                            break
                    nsp_tgt.append(1)
                else:  # 两段来自不同类数据包
                    while True:
                        random_ind_b = random.randint(0, len(data) - 1)
                        if y[random_ind_b] != y[da_ind]:
                            break
                    nsp_tgt.append(0)
                src2 = data[random_ind_b]
                if random.random() < 0.5:
                    random_start = random.randint(0, int(len(src2) / 2))
                else:
                    random_start = 0
                src2 = src2[random_start:]
                src2 = args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize('\t'.join(src2)))[:254 - a_end - 1]
                seg.extend([2] * (len(src2) + 1))

                src.append(args.tokenizer.vocab.get(SEP_TOKEN))
                src.extend(src2)
                src, tgt = create_span_masks(src)
                src.insert(0, args.tokenizer.vocab.get(CLS_TOKEN))
                tgt.insert(0, -1)
                src.append(args.tokenizer.vocab.get(SEP_TOKEN))
                tgt.append(-1)

                while len(src) < args.seq_length:
                    src.append(0)
                    seg.append(0)
                    tgt.append(0)
                ETBert_data.append((src, seg))
                mlm_tgt.append(tgt)

        ETBert_data = np.asarray(ETBert_data)
        mlm_tgt = np.asarray(mlm_tgt)
        nsp_tgt = np.asarray(nsp_tgt)

        # IOS 控制一次加载数据量
        # 获取总数的一半
        n_samples = len(nsp_tgt) // 2
        # 随机选择一半的索引
        indices = np.random.choice(len(nsp_tgt), size=n_samples, replace=False)
        # 使用这些索引选择 data 和 label
        ETBert_data = ETBert_data[indices]
        mlm_tgt = mlm_tgt[indices]
        nsp_tgt = nsp_tgt[indices]

        self.etbert_data = torch.LongTensor(ETBert_data)
        self.mlm_label = torch.LongTensor(mlm_tgt)
        self.nsp_label = torch.LongTensor(nsp_tgt)
        # print(self.data.shape)

    def __len__(self):
        """Summary of __len__.
        
        Returns:
            Any: Description.
        """
        return self.nsp_label.shape[0]  # 返回数据的总个数

    def __getitem__(self, index):
        """Summary of __getitem__.
        
        Args:
            index (Any): Description.
        Returns:
            Any: Description.
        """
        ETBert_data = self.etbert_data[index, :, :]
        mlm_label = self.mlm_label[index]  # 读取每一个npy的数据
        nsp_label = self.nsp_label[index]
        return ETBert_data, mlm_label, nsp_label  # 返回数据还有标签


def prep_dataloader(x_path, y_path, mode, batch_size, njobs=0):
    """Summary of prep_dataloader.
    
    Args:
        x_path (Any): Description.
        y_path (Any): Description.
        mode (Any): Description.
        batch_size (Any): Description.
        njobs (Any): Description.
    Returns:
        Any: Description.
    """
    dataset = ETBERTDataset(x_path, y_path, mode)
    dataloader = DataLoader(dataset, batch_size, shuffle=(mode == 'train'), drop_last=False, num_workers=njobs)
    return dataloader


class MlmTargetGeneration(nn.Module):
    """
    BERT exploits masked language modeling (MLM)
    and next sentence prediction (NSP) for pretraining.
    """

    def __init__(self, args, vocab_size):
        """Summary of __init__.
        
        Args:
            args (Any): Description.
            vocab_size (Any): Description.
        """
        super(MlmTargetGeneration, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = args.hidden_size
        self.emb_size = args.emb_size
        self.factorized_embedding_parameterization = args.factorized_embedding_parameterization
        self.act = str2act[args.hidden_act]

        if self.factorized_embedding_parameterization:
            self.mlm_linear_1 = nn.Linear(args.hidden_size, args.emb_size)
            self.layer_norm = LayerNorm(args.emb_size)
            self.mlm_linear_2 = nn.Linear(args.emb_size, self.vocab_size)
        else:
            self.mlm_linear_1 = nn.Linear(args.hidden_size, args.hidden_size)
            self.layer_norm = LayerNorm(args.hidden_size)
            self.mlm_linear_2 = nn.Linear(args.hidden_size, self.vocab_size)

        self.softmax = nn.LogSoftmax(dim=-1)

        self.criterion = nn.NLLLoss()
        # self.criterion = nn.CrossEntropyLoss()

    def mlm(self, memory_bank, tgt_mlm):
        # Masked language modeling (MLM) with full softmax prediction.
        """Summary of mlm.
        
        Args:
            memory_bank (Any): Description.
            tgt_mlm (Any): Description.
        Returns:
            Any: Description.
        """
        output_mlm = self.act(self.mlm_linear_1(memory_bank))
        output_mlm = self.layer_norm(output_mlm)
        if self.factorized_embedding_parameterization:
            output_mlm = output_mlm.contiguous().view(-1, self.emb_size)
        else:
            output_mlm = output_mlm.contiguous().view(-1, self.hidden_size)
        tgt_mlm = tgt_mlm.contiguous().view(-1)
        output_mlm = output_mlm[tgt_mlm > 0, :]  # 未mask的token均被置0，此处仅选择mask的token计算梯度
        tgt_mlm = tgt_mlm[tgt_mlm > 0]
        output_mlm = self.mlm_linear_2(output_mlm)
        output_mlm = self.softmax(output_mlm)
        # output_mlm = torch.argmax(output_mlm, dim=1)  # 选概率最小的作为对抗样本
        # output_mlm = torch.reshape(output_mlm, (tgt_mlm.shape[0], tgt_mlm.shape[1]))
        # denominator = torch.tensor(output_mlm.size(0) + 1e-6)
        # if output_mlm.size(0) == 0:
        #     correct_mlm = torch.tensor(0.0)
        # else:
        #     correct_mlm = torch.sum((output_mlm.argmax(dim=-1).eq(tgt_mlm)).float())
        loss_mlm = self.criterion(output_mlm, tgt_mlm)
        return loss_mlm, output_mlm

    def mlm_predict(self, memory_bank, tgt_mlm):
        # Masked language modeling (MLM) with full softmax prediction.
        """Summary of mlm_predict.
        
        Args:
            memory_bank (Any): Description.
            tgt_mlm (Any): Description.
        Returns:
            Any: Description.
        """
        output_mlm = self.act(self.mlm_linear_1(memory_bank))
        output_mlm = self.layer_norm(output_mlm)
        if self.factorized_embedding_parameterization:
            output_mlm = output_mlm.contiguous().view(-1, self.emb_size)
        else:
            output_mlm = output_mlm.contiguous().view(-1, self.hidden_size)
        # tgt_mlm = tgt_mlm.contiguous().view(-1)
        # output_mlm = output_mlm[tgt_mlm > 0, :]     # 未mask的token均被置0，此处仅选择mask的token计算梯度
        # tgt_mlm = tgt_mlm[tgt_mlm > 0]
        output_mlm = self.mlm_linear_2(output_mlm)
        output_mlm = self.softmax(output_mlm)
        loss_mlm = None
        return loss_mlm, output_mlm

    def forward(self, memory_bank, tgt, mode='train'):
        """
        Args:
            memory_bank: [batch_size x seq_length x hidden_size]
            tgt: [batch_size x seq_length]

        Returns:
            loss: Masked language modeling loss.
            correct: Number of words that are predicted correctly.
            denominator: Number of masked words.
        """
        if mode == 'train':
            # Masked language model (MLM).
            loss_mlm, output_mlm = self.mlm(memory_bank, tgt)
        else:
            loss_mlm, output_mlm = self.mlm_predict(memory_bank, tgt)

        return loss_mlm, output_mlm


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        """Summary of forward.
        
        Args:
            x (Any): Description.
        Returns:
            Any: Description.
        """
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class SBOTarget(nn.Module):
    def __init__(self, args, vocab_size, max_targets=100, position_embedding_size=768):
        """Summary of __init__.
        
        Args:
            args (Any): Description.
            vocab_size (Any): Description.
            max_targets (Any): Description.
            position_embedding_size (Any): Description.
        """
        super(SBOTarget, self).__init__()
        self.position_embeddings = nn.Embedding(max_targets, position_embedding_size)
        self.liner1 = nn.Linear(args.hidden_size * 3, args.hidden_size, bias=False)
        self.layer_norm1 = BertLayerNorm(args.hidden_size, eps=1e-12)
        self.liner2 = nn.Linear(args.hidden_size, vocab_size, bias=False)
        self.layer_norm2 = BertLayerNorm(vocab_size, eps=1e-12)
        self.act = nn.GELU()
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, memory_bank, src, tgt):
        """Summary of forward.
        
        Args:
            memory_bank (Any): Description.
            src (Any): Description.
            tgt (Any): Description.
        Returns:
            Any: Description.
        """
        sbo_loss = self.compute_sbo_loss(memory_bank, src, tgt)

        return sbo_loss

    def compute_sbo_loss(self, sequence_output, input_ids, masked_spans):

        """Summary of compute_sbo_loss.
        
        Args:
            sequence_output (Any): Description.
            input_ids (Any): Description.
            masked_spans (Any): Description.
        Returns:
            Any: Description.
        """
        sbo_loss = 0
        batch_size, seq_length = masked_spans.shape

        for batch_idx in range(batch_size):
            spans = self.extract_spans(masked_spans[batch_idx])
            if not spans:
                continue

            left_hidden_states = []
            right_hidden_states = []
            targets = []
            position_indices = []

            for span in spans:
                start_idx, end_idx = span[0], span[1]
                if start_idx <= 0 or end_idx >= seq_length - 1:
                    continue

                left_hidden = sequence_output[batch_idx, start_idx - 1, :].unsqueeze(0).repeat(end_idx - start_idx + 1,
                                                                                               1)
                right_hidden = sequence_output[batch_idx, end_idx + 1, :].unsqueeze(0).repeat(end_idx - start_idx + 1,
                                                                                              1)

                left_hidden_states.append(left_hidden)
                right_hidden_states.append(right_hidden)

                for idx in range(start_idx, end_idx + 1):
                    targets.append(input_ids[batch_idx, idx])
                    position_indices.append(idx - start_idx)

            if left_hidden_states and right_hidden_states:
                left_hidden_states = torch.cat(left_hidden_states, dim=0)
                right_hidden_states = torch.cat(right_hidden_states, dim=0)
                targets = torch.tensor(targets, device=sequence_output.device)
                position_indices = torch.tensor(position_indices, device=sequence_output.device)

                position_embeddings = self.position_embeddings(position_indices)
                hidden_states = torch.cat((left_hidden_states, right_hidden_states, position_embeddings), -1)

                hidden_states = self.act(self.layer_norm1(self.liner1(hidden_states)))
                predictions = self.act(self.layer_norm2(self.liner2(hidden_states)))

                sbo_loss += self.loss_fct(predictions, targets)

        return sbo_loss / batch_size

    def extract_spans(self, masked_span):
        """Summary of extract_spans.
        
        Args:
            masked_span (Any): Description.
        Returns:
            Any: Description.
        """
        spans = []
        current_span = None

        for idx, value in enumerate(masked_span):
            if value > 0:
                if current_span is None:
                    current_span = [idx, idx]
                else:
                    current_span[1] = idx
            else:
                if current_span is not None:
                    spans.append(current_span)
                    current_span = None

        if current_span is not None:
            spans.append(current_span)

        return spans


class SpanBERT(pl.LightningModule):
    def __init__(self, args):
        """Summary of __init__.
        
        Args:
            args (Any): Description.
        """
        super(SpanBERT, self).__init__()
        self.save_hyperparameters()
        self.embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
        self.encoder = str2encoder[args.encoder](args)
        self.mlmtarget = MlmTargetGeneration(args, len(args.tokenizer.vocab))
        # self.sbotarget = SBOTarget(args, len(args.tokenizer.vocab))
        self.nsp_linear_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.nsp_linear_2 = nn.Linear(args.hidden_size, 2)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.criterion = nn.NLLLoss()

    def forward(self, src, seg, mlm_y, mode='train'):
        """
        Args:
            src: [batch_size x seq_length]
            seg: [batch_size x seq_length]
        """
        # Embedding.
        emb = self.embedding(src, seg)
        # Encoder.
        output = self.encoder(emb, seg)
        loss_mlm, output_mlm = self.mlmtarget(output, mlm_y, mode)
        # loss_sbo = self.sbotarget(output, src, y)

        output_nsp = torch.tanh(self.nsp_linear_1(output[:, 0, :]))
        output_nsp = self.nsp_linear_2(output_nsp)

        # return loss_mlm+loss_sbo
        return loss_mlm, output_mlm, output_nsp

    def configure_optimizers(self):
        """Summary of configure_optimizers.
        
        Returns:
            Any: Description.
        """
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
        ]
        if args.optimizer in ["adamw"]:
            optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate,
                                                      correct_bias=False)
        else:
            optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate,
                                                      scale_parameter=False, relative_step=False)
        if args.scheduler in ["constant"]:
            scheduler = str2scheduler[args.scheduler](optimizer)
        elif args.scheduler in ["constant_with_warmup"]:
            scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps * args.warmup)
        else:
            scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps * args.warmup, args.train_steps)
        return {'optimizer': optimizer, "lr_scheduler": scheduler}

    def train_dataloader(self):
        """Summary of train_dataloader.
        
        Returns:
            Any: Description.
        """
        tr_set = prep_dataloader(tr_x_path, tr_y_path, mode='train', batch_size=32)
        return tr_set

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        """Summary of training_step.
        
        Args:
            batch (Any): Description.
            batch_idx (Any): Description.
        Returns:
            Any: Description.
        """
        ETBert_data, mlm_y, nsp_y = batch
        mlm_loss, mlm_output, nsp_output = self(ETBert_data[:, 0, :], ETBert_data[:, 1, :], mlm_y)

        nsp_loss = self.criterion(self.softmax(nsp_output), nsp_y)
        train_loss = (mlm_loss / 10) + nsp_loss
        self.log('train_nsp_loss', nsp_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('train_mlm_loss', mlm_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('train_loss', train_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        """Summary of validation_step.
        
        Args:
            batch (Any): Description.
            batch_idx (Any): Description.
        """
        ETBert_data, mlm_y, nsp_y = batch
        mlm_loss, mlm_output, nsp_output = self(ETBert_data[:, 0, :], ETBert_data[:, 1, :], mlm_y)

        nsp_loss = self.criterion(self.softmax(nsp_output), nsp_y)
        val_loss = (mlm_loss / 10) + nsp_loss
        self.log('val_nsp_loss', nsp_loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log('val_mlm_loss', mlm_loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log('val_loss', val_loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)


checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    filename=config['save_name'],
    save_top_k=1,
    mode='min',
    save_last=True
)

if __name__ == "__main__":
    device = get_device()
    val_set = prep_dataloader(val_x_path, val_y_path, mode='val', batch_size=32)
    # instances_num = len(val_set.dataset)
    instances_num = 4152
    args.train_steps = int(instances_num * config['n_epochs'] / config['batch_size']) + 1
    model = SpanBERT(args)
    model.load_state_dict(torch.load(args.pretrained_model_path,
                                     map_location={'cuda:1': 'cuda:0', 'cuda:2': 'cuda:0',
                                                   'cuda:3': 'cuda:0'}),
                          strict=False)

    logger = TensorBoardLogger(config['log_path1'], config['log_path2'])
    trainer = Trainer(val_check_interval=1.0, max_epochs=config['n_epochs'], devices='auto', accelerator='auto',
                      logger=logger, reload_dataloaders_every_n_epochs=1,
                      # 每epoch重新引入dataloader      # fast_dev_run=True,
                      callbacks=[EarlyStopping(monitor='val_loss', mode='min', check_on_train_epoch_end=True),
                                 checkpoint_callback])
    trainer.fit(model, val_dataloaders=val_set)

    # test_set = prep_dataloader(test_x_path, test_y_path, 'test', config['batch_size'], njobs=1)
    # trainer.test(model, dataloaders=test_set)
