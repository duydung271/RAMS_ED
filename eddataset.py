import transformers
from torch.utils.data import Dataset, DataLoader
import json
import torch


def load_json(path):
    with open(path) as f:
        data = json.load(f)
    return data


def load_sentence_data(path):
    data = []
    print('| Loading: ', path)
    with open(path) as f:
        lines = f.read().split('\n')

    for line in lines:
        line = line.strip()
        if len(line) == 0:
            continue
        doc = json.loads(line)
        sentences = doc['sentences']
        triggers = doc['evt_triggers']

        min_accept = -1
        max_accept = 0
        for words in sentences:
            sent_labels = ['O' for _ in words]
            max_accept += len(words)

            for trigger in triggers:
                # print(trigger)
                doc_start, doc_end, label = trigger
                label = label[0][0]
                # print(min_accept, max_accept, doc_start)
                if min_accept < doc_start < max_accept:
                    start = doc_start - min_accept - 1
                    end = doc_end - min_accept - 1
                    assert 0 <= start <= end < len(sent_labels)
                    sent_labels[start] = 'B-' + label
                    for i in range(start + 1, end + 1):
                        sent_labels[start] = 'I-' + label
            min_accept += len(words)
            if len(words) > 200:
                continue
            data.append((words, sent_labels))
    return data


class EDDataset(Dataset):

    def __init__(self, path, label2index, tokenizer, args):
        super(EDDataset, self).__init__()
        self.sentence_data = load_sentence_data(path)
        self.label2index = label2index
        self.tokenizer = tokenizer
        self.CLS = self.tokenizer.cls_token_id
        self.PAD = self.tokenizer.pad_token_id
        self.SEP = self.tokenizer.sep_token_id
        self.UNK = self.tokenizer.unk_token_id
        # BERT MAX LEN
        self.BML = 512
        # Word MAX LEN
        self.WML = 386

    def __len__(self):
        return len(self.sentence_data)

    def __getitem__(self, item):
        words, labels = self.sentence_data[item]
        word_length = len(words)

        all_pieces = [self.CLS]
        all_spans = []
        targets = [self.label2index[x] for x in labels]

        for word in words:
            tokens = self.tokenizer.tokenize(word)
            pieces = self.tokenizer.convert_tokens_to_ids(tokens)
            if len(pieces) == 0:
                pieces = [self.UNK]
            start = len(all_pieces)
            all_pieces += pieces
            end = len(all_pieces)
            all_spans.append([start, end])
        all_pieces.append(self.SEP)
        cls_text_sep_length = len(all_pieces)

        assert len(all_pieces) <= self.BML
        pad_len = self.BML - len(all_pieces)
        all_pieces += [self.PAD] * pad_len
        attention_mask = [1.0] * cls_text_sep_length + [0.0] * pad_len
        assert len(all_pieces) == self.BML

        return {
            'words': words,
            'labels': labels,
            'indices': all_pieces,
            'attention_mask': attention_mask,
            'word_spans': all_spans,
            'bert_length': cls_text_sep_length,
            'word_length': word_length,
            'target': targets
        }

    @staticmethod
    def pack(items):
        return {
            k: TS_TYPE([x[k] for x in items])
            for k, TS_TYPE in TENSOR_TYPES.items()
        }


def keep(items):
    return items


def flatten(items):
    all_items =  [y for x in items for y in x]
    return torch.LongTensor(all_items)


TENSOR_TYPES = {
    'words': keep,
    'labels': keep,
    'indices': torch.LongTensor,
    'attention_mask': torch.FloatTensor,
    'word_spans': keep,
    'bert_length': torch.LongTensor,
    'word_length': torch.LongTensor,
    'target': flatten
}

if __name__ == '__main__':
    tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-cased')
    args = None
    train = load_sentence_data('data/train.jsonlines')
    dev = load_sentence_data('data/dev.jsonlines')
    test = load_sentence_data('data/test.jsonlines')
    all_data = train + dev + test

    label_set = set()
    WML = 0
    for x in train + dev + test:
        label_set.update(x[1])
        WML = max(WML, len(x[0]))
    label_set.remove('O')
    labels = ['O'] + sorted(label_set)
    label2index = {
        x: i for i, x in enumerate(labels)
    }

    with open('data/label_mapping.json', 'w') as f:
        json.dump(label2index, f, indent=2)
    print('Word_max_length:', WML)

    # d = EDDataset('data/test.jsonlines', label2index, tokenizer, args)
    # for x in d:
    #     print(x)
