import random
import sys
import os
import torch
import argparse
import collections
import torch.nn as nn

uer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print(uer_dir)
sys.path.append(uer_dir)

from uer.utils.constants import *
from uer.utils import *
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.model_loader import load_model
from uer.opts import infer_opts

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
from adv_fine_tuning_cls import SpanBERT

import time
import scapy
from scapy.all import *
from scapy.layers.inet import IP

# packet
test_x_path = r'D:\dataset_packet\vpn_dataset_split\traffic_classification\test\x_bert.pickle'
test_y_path = r'D:\dataset_packet\vpn_dataset_split\traffic_classification\test\y.npy'
save_path = r'D:\dataset_packet\vpn_dataset_split\traffic_classification\test\\'

# TSCRNN
# test_x_path = r'D:\dataset_TSCRNN\ustc_dataset\x_bert.pickle'
# test_y_path = r'D:\dataset_TSCRNN\ustc_dataset\y_test_sorted.npy'
# save_path = r'D:\dataset_TSCRNN\ustc_dataset\\'
fun = 'fun5'  # fun1: 一个seg，argmin;
#               fun2：两个seg，argmax;
#               fun3：两个seg，前argmin，后argmax，各1/2;
#               fun4:两个seg，前argmin，后argmax，分布随机；
#               random；
#               fun1_argmax：两个seg，都用argmax
#               fun5: dummy head, 两段随机
#               fun6: cls决定 argmax or argmin
#               random_dummy_head
#               random_pad
BWO = 0.1

target = 'mlm'
# --load_model_path D:/project/ET-BERT/ET-BERT-main/models/pretrained_model.bin \
# --load_model_path D:\\project\\ET-BERT\\ET-BERT-main\\packet_adv\\packet_adv_span_2seg\\vpn\\version_1\\checkpoints\\last.ckpt \
# --load_model_path D:\\project\\ET-BERT\\ET-BERT-main\\packet_adv\\packet_adv_span_2seg\\ios\\version_1\\checkpoints\\last.ckpt \
# --load_model_path D:\\project\\ET-BERT\\ET-BERT-main\\packet_adv\\packet_adv_span_2seg\\tor\\version_0\\checkpoints\\last.ckpt \
# --load_model_path D:\\project\\ET-BERT\\ET-BERT-main\\packet_adv\\model\\vpn.ckpt \
# --load_model_path D:\\project\ET-BERT\\ET-BERT-main\\packet_adv\\span_2seg_cls_log\\vpn_2e-5_30%\\version_1\\checkpoints\\vpn_span-epoch=49-val_loss=1.52.ckpt \
parameter = '''
--load_model_path D:\\project\\ET-BERT\\ET-BERT-main\\packet_adv\\model\\vpn.ckpt \
--vocab_path D:/project/ET-BERT/ET-BERT-main/models/encryptd_vocab.txt \
--embedding word_pos_seg 
--encoder transformer 
--mask fully_visible
--test_path none, --prediction_path none, --labels_num 0
--seq_length 256
'''

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

infer_opts(parser)

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

# Build tokenizer.
args.tokenizer = str2tokenizer[args.tokenizer](args)

# Build classification model and load parameters.
args.soft_targets, args.soft_alpha = False, False

prefix = []
tail = []


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
        # getPcap(data, "tor.pcap")
        # packet
        y = np.load(save_path + 'y.npy')
        # TSCRNN
        # y = np.load(save_path + 'y_test_sorted.npy')
        # y = np.repeat(y, 15)

        # inser_pos = np.load(save_path+'insert_start_grad.npy')
        ETBert_data = []
        mlm_tgt = []
        insert_ind = 20  # 插入扰动位置，40字节处（双字节20）为head结尾payload开头处
        for ind, item in tqdm(enumerate(data)):
            # insert_ind = inser_pos[ind] // 2
            # if len(ETBert_data) == 128: break  # 限定读取数量，debug使用
            # mask_num = min(int(0.1 * len(item)), 50)
            if len(item)==1: # TSCRNN pad data
                src = [0] * 256
                tgt = [0] * 256
                seg = [1] * 256
                ETBert_data.append((src, seg))
                mlm_tgt.append(tgt)
                prefix.append(None)
                tail.append(None)
                continue
            mask_num = int(BWO * len(item))
            if fun == 'fun5' or fun == 'fun6':
                mask_num = mask_num + 20
                # if mask_num<20:
                #     mask_num = 20

            max_subseq_length = 250 - mask_num
            start = max(0, insert_ind - max_subseq_length // 2)
            end = min(len(item), start + max_subseq_length)

            # 确保子序列的中心尽量接近插入位置
            if end - start < max_subseq_length:
                start = max(0, end - max_subseq_length)

            if start > 0:
                prefix.append(item[:start])
            else:
                prefix.append(None)
            if end < len(item):
                tail.append(item[end:])
            else:
                tail.append(None)

            item = item[start:end]
            relative_insert_pos = insert_ind - start

            before_mask = item[:relative_insert_pos]
            after_mask = item[relative_insert_pos:]
            before_mask = args.tokenizer.tokenize('\t'.join(before_mask))
            after_mask = args.tokenizer.tokenize('\t'.join(after_mask))
            if fun == 'fun1' or fun == 'fun1_argmax':
                src = args.tokenizer.convert_tokens_to_ids(
                    [CLS_TOKEN] + before_mask + [MASK_TOKEN] * mask_num + after_mask + [SEP_TOKEN])
                tgt = [0] + [0] * len(before_mask) + [1] * mask_num + [0] * len(after_mask) + [0]
                seg = [1] * len(src)
            elif fun == 'fun2':
                src = args.tokenizer.convert_tokens_to_ids(
                    [CLS_TOKEN] + before_mask + [SEP_TOKEN] + [MASK_TOKEN] * mask_num + after_mask)
                tgt = [0] + [0] * len(before_mask) + [0] + [1] * mask_num + [0] * len(after_mask)
                seg = [1] * (len(before_mask) + 2) + [2] * (mask_num + len(after_mask))
            elif fun == 'fun3':
                src = args.tokenizer.convert_tokens_to_ids(
                    [CLS_TOKEN] + before_mask + [MASK_TOKEN] * int(mask_num / 2) + [SEP_TOKEN] + [MASK_TOKEN] * int(
                        mask_num / 2) + after_mask)
                tgt = [0] + [0] * len(before_mask) + [1] * int(mask_num / 2) + [0] + [1] * int(mask_num / 2) + [
                    0] * len(after_mask)
                seg = [1] * (len(before_mask) + 2 + int(mask_num / 2)) + [2] * (int(mask_num / 2) + len(after_mask))
            elif fun == 'fun4':
                if mask_num > 2:
                    seg1 = random.randint(1, mask_num - 1)
                    seg2 = mask_num - seg1
                else:
                    seg1 = int(mask_num / 2)
                    seg2 = int(mask_num / 2)
                src = args.tokenizer.convert_tokens_to_ids(
                    [CLS_TOKEN] + before_mask + [MASK_TOKEN] * seg1 + [SEP_TOKEN] + [MASK_TOKEN] * seg2 + after_mask)
                tgt = [0] + [0] * len(before_mask) + [1] * seg1 + [0] + [1] * seg2 + [0] * len(after_mask)
                seg = [1] * (len(before_mask) + 2 + seg1) + [2] * (seg2 + len(after_mask))
            elif fun == 'fun5' or fun == 'fun6':
                mask_num = mask_num - 20
                while True:
                    data_ind = np.random.randint(0, len(data) - 1)
                    if y[data_ind] != y[ind]:
                        break
                if mask_num > 2:
                    seg1 = random.randint(1, mask_num - 1)
                    seg2 = mask_num - seg1
                else:
                    seg1 = mask_num
                    seg2 = 0
                # seg1 = mask_num // 2  # 各一半
                # seg2 = mask_num - seg1

                # seg1 = mask_num  #  动态调节
                # seg2 = 0

                head2 = args.tokenizer.tokenize('\t'.join(data[data_ind][:20]))
                src = args.tokenizer.convert_tokens_to_ids(
                    [CLS_TOKEN] + before_mask + [MASK_TOKEN] * seg1 + [SEP_TOKEN] + head2 + [
                        MASK_TOKEN] * seg2 + after_mask)
                tgt = [0] + [0] * len(before_mask) + [1] * seg1 + [0] * (1 + len(head2)) + [1] * seg2 + [0] * len(
                    after_mask)  # dummy head不作为预测目标
                # tgt = [0] + [0] * len(before_mask) + [1] * seg1 + [0] * 1+[1] * (seg2++len(head2)) + [0] * len(after_mask)      # dummy head作为预测目标，效果差
                seg = [1] * (len(before_mask) + 2 + seg1) + [2] * (seg2 + len(after_mask) + len(head2))
                # seg = [1] * (len(before_mask)+2+seg1) + [1] * len(head2) + [2] * (seg2+len(after_mask))    # dummy head 算作分段1，按argmin
            if len(src) > args.seq_length:
                tail_add = src[args.seq_length:]  # 分词后长度超出，将超出部分保存以待复原数据包
                tail_add = args.tokenizer.convert_ids_to_tokens(tail_add)
                if "#" in tail_add[0]:
                    tail_add = src[args.seq_length - 1:]
                    tail_add = args.tokenizer.convert_ids_to_tokens(tail_add)
                    src = src[: args.seq_length - 1]
                    seg = seg[: args.seq_length - 1]
                    tgt = tgt[: args.seq_length - 1]
                else:
                    src = src[: args.seq_length]
                    seg = seg[: args.seq_length]
                    tgt = tgt[: args.seq_length]

                tail_add = tokens_to_text(tail_add)
                if tail[-1] is None:
                    tail[-1] = tail_add
                else:
                    tail[-1] = tail_add + tail[-1]

            while len(src) < args.seq_length:
                src.append(0)
                seg.append(0)
                tgt.append(0)
            ETBert_data.append((src, seg))
            mlm_tgt.append(tgt)
        self.etbert_data = torch.LongTensor(np.asarray(ETBert_data))
        self.label = torch.LongTensor(np.asarray(mlm_tgt))
        # print(self.data.shape)

    def __len__(self):
        """Summary of __len__.
        
        Returns:
            Any: Description.
        """
        return self.label.shape[0]  # 返回数据的总个数

    def __getitem__(self, index):
        """Summary of __getitem__.
        
        Args:
            index (Any): Description.
        Returns:
            Any: Description.
        """
        ETBert_data = self.etbert_data[index, :, :]
        label = self.label[index]  # 读取每一个npy的数据
        return ETBert_data, label  # 返回数据还有标签


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


def load_BERT():
    # model = SpanBERT(args)
    """Summary of load_BERT.
    
    Returns:
        Any: Description.
    """
    model = SpanBERT.load_from_checkpoint(args.load_model_path)
    # model = model.load_state_dict(torch.load(args.load_model_path, map_location="cpu"), strict=True)

    # For simplicity, we use DataParallel wrapper to use multiple GPUs.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    return model


def tokens_to_text(tokens):
    """Summary of tokens_to_text.
    
    Args:
        tokens (Any): Description.
    Returns:
        Any: Description.
    """
    words = []
    for token in tokens:
        if token in ['[CLS]', '[SEP]', '[PAD]', '[UNK]', '[MASK]']:  # 过滤特殊token
            continue
        if token.startswith('##'):
            words[-1] = words[-1] + token[2:]  # 去掉'##'并拼接到前一个词上
        else:
            words.append(token)
    return words
    # return ' '.join(words)


def process_tensor(x, seg, nsp):
    """Summary of process_tensor.
    
    Args:
        x (Any): Description.
        seg (Any): Description.
        nsp (Any): Description.
    Returns:
        Any: Description.
    """
    batch, seqLen, embDim = x.shape
    result = torch.zeros(batch, seqLen, dtype=torch.long).cuda()

    for i in range(batch):
        sentence1_mask = seg[i] == 1
        sentence2_mask = seg[i] == 2

        # 句子1部分的处理
        if sentence1_mask.any():
            sentence1_indices = torch.where(sentence1_mask)[0]
            sentence1_values = x[i, sentence1_indices]
            min_indices = sentence1_values.argmin(dim=1)
            result[i, sentence1_indices] = min_indices
        # 句子2部分的处理
        if sentence2_mask.any():
            sentence2_indices = torch.where(sentence2_mask)[0]
            sentence2_values = x[i, sentence2_indices]
            if fun == 'fun6' and nsp[i] == 1:
                max_indices = sentence2_values.argmin(dim=1)
            else:
                max_indices = sentence2_values.argmax(dim=1)
            result[i, sentence2_indices] = max_indices

    return result


def delete_head(mask, row_data):
    """Summary of delete_head.
    
    Args:
        mask (Any): Description.
        row_data (Any): Description.
    Returns:
        Any: Description.
    """
    for i in range(mask.shape[0]):
        first_one = np.argmax(mask)
        last_one = len(mask) - 1 - np.argmax(mask[::-1])
        idx = 500
        for i in range(len(row_data)):
            if row_data[i] == '[SEP]':
                idx = i
                break
        if idx < first_one: first_one = idx

        # 创建新的行，只保留 mask 为 1 的元素
        new_row = [row_data[j] for j in range(len(mask)) if mask[j] == 1 or j < first_one or j > last_one]

        return new_row


def head_random(mask, row_data):  # 改dummy head IP地址端口号为随机
    """Summary of head_random.
    
    Args:
        mask (Any): Description.
        row_data (Any): Description.
    Returns:
        Any: Description.
    """
    first_one = np.argmax(mask)
    second_one = len(mask) - 1 - np.argmax(mask[::-1])
    # 提取两段 1 之间 0 对应的序列
    middle_zeros_indices = \
    np.where((mask == 0) & (np.arange(len(mask)) > first_one) & (np.arange(len(mask)) < second_one))[0]

    # 修改该段序列中位置为 7-14 位的值为随机值
    if len(middle_zeros_indices) > 0:
        for j in middle_zeros_indices:
            if row_data[j] == '0000':
                random_ids = np.random.randint(len(args.tokenizer.vocab), size=1)
                row_data[j] = args.tokenizer.convert_ids_to_tokens(random_ids)[0]
    return row_data


def generate_sample(model, data):
    """Summary of generate_sample.
    
    Args:
        model (Any): Description.
        data (Any): Description.
    """
    model.eval()
    x_adv = []
    with torch.no_grad():
        time_consumpt = []
        for x, y in tqdm(test_set):  # y为mlm_tgt
            start = time.time()
            # os.system("nvidia-smi --query-gpu=memory.used --format=csv")
            loss_mlm, output_mlm, output_nsp = model(x[:, 0, :].cuda(), x[:, 1, :].cuda(), y.cuda(), mode='test')  # 0.009772021561036185
            # os.system("nvidia-smi --query-gpu=memory.used --format=csv")
            nsp = torch.argmax(output_nsp, dim=1)  # 1:同一类； 0：不同类
            # end = time.time()
            if fun == 'fun1':
                output_mlm = torch.argmin(output_mlm, dim=1)  # 选概率最小的作为对抗样本
                y_gen = torch.reshape(output_mlm, (y.shape[0], y.shape[1]))
            elif fun == 'fun2':
                output_mlm = torch.argmax(output_mlm, dim=1)  # 选概率最大的作为对抗样本
                y_gen = torch.reshape(output_mlm, (y.shape[0], y.shape[1]))
            elif fun == 'fun3' or fun == 'fun4' or fun == 'fun5' or fun == 'fun6':
                output_mlm = torch.reshape(output_mlm, (y.shape[0], y.shape[1], len(args.tokenizer.vocab)))
                y_gen = process_tensor(output_mlm, x[:, 1, :].cuda(), nsp)
            elif fun == 'fun1_argmax':
                output_mlm = torch.argmax(output_mlm, dim=1)
                y_gen = torch.reshape(output_mlm, (y.shape[0], y.shape[1]))
            end = time.time()
            time_consumpt.append(end-start)
            x = x[:, 0, :].cpu().numpy()
            y_gen = y_gen.cpu().numpy()
            x[y == 1] = y_gen[y == 1]
            y = y.cpu().numpy()
            for idx, item in enumerate(x):
                # last_non_zero = np.nonzero(item)[0][-1]
                # item = item[1:last_non_zero]     # 删去开头的cls 结尾的sep与pad (合并至tokens_to_text)
                sample = args.tokenizer.convert_ids_to_tokens(item)
                # if fun == 'fun5':
                #     sample = delete_head(y[idx], sample)
                #     sample = head_random(y[idx], sample)
                sample = tokens_to_text(sample)
                x_adv.append(sample)
    start = time.time()
    for i in range(len(x_adv)):
        if prefix[i] is not None:
            x_adv[i] = prefix[i] + x_adv[i]
        if tail[i] is not None:
            x_adv[i] = x_adv[i] + tail[i]
    print(np.average(time_consumpt))
    # 获取生成对抗数据包
    # getPcap(x_adv, "tor_PacketPatch.pcap")
    # end = time.time()
    # all_time = end-start
    # print(all_time, all_time/ 29414)
    # with open(save_path + 'x_adv_span_head_' + fun + str(BWO)+ '.pickle', 'wb') as file:
    #     pickle.dump(x_adv, file)

    deep_packet_data = []
    for item in tqdm(x_adv):
        decimal_list = []
        for hex_str in item:
            # 将每两个字符的16进制数转换回十进制
            for i in range(0, len(hex_str), 2):
                byte_str = hex_str[i:i + 2]
                decimal_value = int(byte_str, 16)
                decimal_list.append(decimal_value)
        if len(decimal_list) < 1500:
            decimal_list = decimal_list + [0] * (1500 - len(decimal_list))
            decimal_list = [x / 255 for x in decimal_list]
        elif len(decimal_list) > 1500:
            decimal_list = decimal_list[:1500]
            decimal_list = [x / 255 for x in decimal_list]
        deep_packet_data.append(decimal_list)
    deep_packet_data = np.asarray(deep_packet_data)
    # TSCRNN
    # deep_packet_data = deep_packet_data.reshape(int(deep_packet_data.shape[0]/15), 15, 1500)
    # np.save(save_path + 'x_adv_span_head_deep_packet_' + fun + str(BWO)+ '.npy', deep_packet_data)


def random_like_bert():
        """Summary of random_like_bert.
        """
        with open(test_x_path, 'rb') as file:
            data = pickle.load(file)
        # inser_pos = np.load(save_path+'insert_start.npy')
        x_adv = []
        insert_ind = 20  # 插入扰动位置，40字节处（双字节20）为head结尾payload开头处
        for ind, item in tqdm(enumerate(data)):
            # insert_ind = inser_pos[ind] // 2
            # if len(ETBert_data) == 128: break  # 限定读取数量，debug使用
            # mask_num = min(int(0.1 * len(item)), 50)
            mask_num = max(int(0.1 * len(item)), 1)
            # if mask_num < 20:
            #     mask_num = 20
            while True:
                sampled_indices = np.random.randint(len(args.tokenizer.vocab), size=mask_num)
                sample = args.tokenizer.convert_ids_to_tokens(sampled_indices)
                if '#' not in sample[0]:
                    break
            sample = tokens_to_text(sample)
            t = item[:20] + sample + item[20:]
            x_adv.append(t)
        deep_packet_data = []
        for item in tqdm(x_adv):
            decimal_list = []
            for hex_str in item:
                # 将每两个字符的16进制数转换回十进制
                for i in range(0, len(hex_str), 2):
                    byte_str = hex_str[i:i + 2]
                    decimal_value = int(byte_str, 16)
                    decimal_list.append(decimal_value)
            if len(decimal_list) < 1500:
                decimal_list = decimal_list + [0] * (1500 - len(decimal_list))
                decimal_list = [x / 255 for x in decimal_list]
            elif len(decimal_list) > 1500:
                decimal_list = decimal_list[:1500]
            deep_packet_data.append(decimal_list)
        deep_packet_data = np.asarray(deep_packet_data)
        np.save(save_path + 'x_adv_random.npy', deep_packet_data)


def random_pad():
    """Summary of random_pad.
    """
    global test_x_path
    test_x_path = test_x_path.replace("x_bert.pickle", "x_without_pad.pickle")
    with open(test_x_path, 'rb') as file:
        data = pickle.load(file)
    y = np.load(save_path + 'y.npy')
    # inser_pos = np.load(save_path+'insert_start.npy')
    deep_packet = []
    bert_data = []
    insert_ind = 40  # 插入扰动位置，40字节处（双字节20）为head结尾payload开头处
    for ind, item in tqdm(enumerate(data)):
        pad = []
        # pad_num = int(0.08*len(item))
        pad_num = 40
        if pad_num+len(item)>1500:
            pad_num = 1500-len(item)
        for i in range(pad_num):  # IP地址，端口号
            pad.append(random.randint(0, 256))
        item = item[:insert_ind] + pad + item[insert_ind:]

        hex_data = [f'{x:02x}' for x in item]
        if len(hex_data) % 2 != 0:
            hex_data.append('00')
        combined_result = [hex_data[i] + hex_data[i + 1] for i in range(0, len(hex_data), 2)]
        bert_data.append(combined_result)

        if len(item) > 1500:
            item = item[:1500]
        while len(item) < 1500:
            item.append(0)
        deep_packet.append(item)

    with open(save_path + 'x_random_pad_40B' + '.pickle', 'wb') as file:
        pickle.dump(bert_data, file)

    deep_packet = np.asarray(deep_packet)
    deep_packet = deep_packet / 255
    np.save(save_path + 'x_random_pad_40B.npy', deep_packet)


def random_dummy_head():
    # np.random.seed(10)
    """Summary of random_dummy_head.
    """
    global test_x_path
    test_x_path = test_x_path.replace("x_bert.pickle", "x_without_pad.pickle")
    with open(test_x_path, 'rb') as file:
        data = pickle.load(file)
    y = np.load(save_path + 'y.npy')
    # inser_pos = np.load(save_path+'insert_start.npy')
    deep_packet = []
    bert_data = []
    insert_ind = 40  # 插入扰动位置，40字节处（双字节20）为head结尾payload开头处
    for ind, item in tqdm(enumerate(data)):
        while True:
            data_ind = np.random.randint(0, len(data) - 1)
            if y[data_ind] != y[ind] and len(data[data_ind])>40:
                break
        head2 = data[data_ind][:40]
        # for i in range(12,22):# IP地址，端口号
        #     head2[i] = random.randint(0,256)
        item = item[:insert_ind] + head2 + item[insert_ind:]

        hex_data = [f'{x:02x}' for x in item]
        if len(hex_data) % 2 != 0:
            hex_data.append('00')
        combined_result = [hex_data[i] + hex_data[i + 1] for i in range(0, len(hex_data), 2)]
        bert_data.append(combined_result)

        if len(item)>1500:
            item = item[:1500]
        while len(item)<1500:
            item.append(0)
        deep_packet.append(item)

    # with open(save_path + 'x_dummyhead_IPTCP' + '.pickle', 'wb') as file:
        pickle.dump(bert_data, file)

    deep_packet = np.asarray(deep_packet)
    deep_packet = deep_packet / 255
    # np.save(save_path + 'x_dummyhead_IPTCP.npy', deep_packet)


def getPcap(data, name):
    """Summary of getPcap.
    
    Args:
        data (Any): Description.
        name (Any): Description.
    Returns:
        Any: Description.
    """
    pkts = []
    for item in tqdm(data):
        decimal_list = []
        for hex_str in item:
            # 将每两个字符的16进制数转换回十进制
            for i in range(0, len(hex_str), 2):
                byte_str = hex_str[i:i + 2]
                decimal_value = int(byte_str, 16)
                decimal_list.append(decimal_value)
        pkt = bytearray(decimal_list)
        pkt = IP(pkt)
        pkts.append(pkt)
    wrpcap(name, pkts)
    return data

if __name__ == "__main__":
    if fun == 'random':
        random_like_bert()
    elif fun == 'random_dummy_head':
        random_dummy_head()
    elif fun == 'random_pad':
        random_pad()
    else:
        os.system("nvidia-smi --query-gpu=memory.used --format=csv")
        test_set = prep_dataloader(test_x_path, test_y_path, mode='test', batch_size=1)
        model = load_BERT()
        generate_sample(model, test_set)
        # # 选择设备
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # # 清除缓存，确保统计精确
        # torch.cuda.reset_peak_memory_stats(device)
        # torch.cuda.empty_cache()
        #
        # # 记录初始显存占用
        # start_mem = torch.cuda.memory_allocated(device)
        # # 记录推理后的显存占用
        # end_mem = torch.cuda.memory_allocated(device)
        # max_mem = torch.cuda.max_memory_allocated(device)
        #
        # print(f"推理前显存: {start_mem / 1024 ** 2:.2f} MB")
        # print(f"推理后显存: {end_mem / 1024 ** 2:.2f} MB")
        # print(f"推理过程中最大显存: {max_mem / 1024 ** 2:.2f} MB")
        # os.system("nvidia-smi --query-gpu=memory.used --format=csv")
