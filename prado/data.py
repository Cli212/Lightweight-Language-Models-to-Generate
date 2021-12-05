import torch
import itertools
from itertools import chain, repeat, islice
import numpy as np
from typing import Any, Dict, List, Tuple, Union
from sklearn.model_selection import train_test_split
from torch import Tensor, BoolTensor
from torch.utils.data import Dataset, DataLoader
from .hash import murmurhash


def token_hash(tokens, feature_size):
    hashings = []
    for j in range(len(tokens)):
        curr_hashing = murmurhash(
            tokens[j], feature_size=feature_size
        )
        hashings.append(curr_hashing[: feature_size // 2])
    return hashings


def zeroPadding(iterable, max_length, PAD_token=None):
    def pad_infinite(iterable, PAD_token=None):
        return chain(iterable, repeat(PAD_token))

    return list(islice(pad_infinite(iterable, PAD_token), max_length))


def binaryMatrix(l, PAD_token):
    m = []
    # for i, seq in enumerate(l):
    #     m.append([])
    for token in l:
        if token == PAD_token:
            m.append(0)
        else:
            m.append(1)
    return m


class Voc:
    def __init__(self, vocab_path, PAD_token, SOS_token, EOS_token):
        with open(vocab_path) as f:
            word_list = [(i + 3, w.strip()) for i, w in enumerate(f.readlines())]
            self.index2word = dict(word_list)
        self.index2word[PAD_token] = "PAD"
        self.index2word[SOS_token] = "SOS"
        self.index2word[EOS_token] = "EOS"
        self.SOS_token = SOS_token
        self.PAD_token = PAD_token
        self.EOS_token = EOS_token
        self.word2index = dict(zip(self.index2word.values(), self.index2word.keys()))
        self.num_words = len(self.word2index)


class QADataset(Dataset):
    """
    A dataset object that store question, answer and mask tensor
    """
    def __init__(self, data: List[Tuple[List[Any], List[Any]]], pad, sos, eos, max_length, proj_feature_size=None,
                 voc=None):
        self.sos_tag = sos
        self.eos_tag = eos
        self.pad = pad
        self.max_length = max_length
        self.data = data
        self.proj_feature_size = proj_feature_size
        self.voc = voc

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        question, answer = zeroPadding([self.sos_tag] + self.data[item][0][:self.max_length - 2] + [self.eos_tag],
                                       self.max_length, self.pad), zeroPadding(
            [self.sos_tag] + self.data[item][1][:self.max_length - 2] + [self.eos_tag], self.max_length, self.pad)
        mask = binaryMatrix(answer, self.pad)
        length = min(len(self.data[item][0]) + 2, self.max_length)
        if self.proj_feature_size:
            q_tokens = [self.voc.index2word[i] for i in question]
            q_hashings = token_hash(q_tokens, self.proj_feature_size)
            return q_hashings, answer, mask, length
        else:
            return question, answer, mask, length


def collate_fn(examples: List[Any]) -> Tuple[Any, Any, Tensor, BoolTensor, int]:
    """Batching examples.
    Parameters
    ----------
    examples : List[Any]
        List of examples
    Returns
    -------
    Tuple[torch.Tensor, ...]
        Tuple of hash tensor, length tensor, and label tensor
    """

    projection = []
    labels = []
    masks = []
    lengths = []
    for example in examples:
        if not isinstance(example, tuple):
            projection.append(np.asarray(example))
        else:
            projection.append(np.asarray(example[0]))
            labels.append(example[1])
            masks.append(example[2])
            lengths.append(example[3])
    lengths = torch.from_numpy(np.asarray(lengths)).long()
    masks = torch.BoolTensor(masks)
    max_target_len = max([len(indexes) for indexes in labels])
    try:
        projection_tensor = np.zeros(
            (len(projection), max(map(len, projection)), len(projection[0][0]))
        )
        for i, doc in enumerate(projection):
            projection_tensor[i, : len(doc), :] = doc
        return (
            torch.from_numpy(projection_tensor).float(),
            lengths,
            torch.from_numpy(np.asarray(labels)),
            masks,
            max_target_len
        )
    except:
        projection_tensor = np.asarray(projection)
        return (
            torch.from_numpy(projection_tensor).long(),
            lengths,
            torch.from_numpy(np.asarray(labels)),
            masks,
            max_target_len
        )


def create_dataloaders(df, sos, eos, pad, max_length, batch_size, train_size=0.8, proj_feature_size=None, voc=None,
                       seed=123):
    """
    Create train and validation dataloaders from a dataframe with two colums "question index" and "answer index"
    :param df: a dataframe including questions and answers as the list of tokens of them
    :param sos: start of sentence token
    :param eos: end of sentence token
    :param pad: padding token
    :param max_length: max sequence length
    :param batch_size:
    :param train_size:
    :param proj_feature_size: if use projection to build embedding, then the proj_feature_size should be given
    :param voc:
    :param seed: random seed, fix to reproduce the result
    :return:
    """
    q_tokens = [eval(i) for i in df['question_index'].values.tolist()]
    a_tokens = [eval(i) for i in df['answer_index'].values.tolist()]
    q_tokens = [[j + 3 for j in i] for i in q_tokens]
    a_tokens = [[j + 3 for j in i] for i in a_tokens]
    train_tokens, val_tokens = train_test_split(list(zip(q_tokens, a_tokens)), train_size=train_size, random_state=seed)
    train_dataset = QADataset(train_tokens, sos, eos, pad, max_length, proj_feature_size=proj_feature_size, voc=voc)
    val_dataset = QADataset(val_tokens, sos, eos, pad, max_length, proj_feature_size=proj_feature_size, voc=voc)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=1,
        drop_last=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=1,
        drop_last=True
    )
    return (train_dataloader, val_dataloader)


def create_dataloader_from_sentence(index_batch, voc, max_length, proj_feature_size):
    """
    Creat a dataloader for the test mode (1 sentence at a time)
    :param index_batch: a batch with size [batch_size, max_seq_length]
    :param voc: vocabulary object
    :param max_length:
    :param proj_feature_size:
    :return:
    """
    test_dataset = QADataset(index_batch, voc.SOS_token, voc.EOS_token, voc.PAD_token, max_length,
                             proj_feature_size=proj_feature_size, voc=voc)
    return DataLoader(
        test_dataset,
        batch_size=1,
        collate_fn=collate_fn,
        num_workers=1,
        drop_last=False
    )
