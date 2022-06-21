import argparse

import torch.optim
import tqdm
import os

from model import *
from eddataset import *
from transformers import AutoTokenizer
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, recall_score, f1_score


def arugment_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--update_bert', default=False, action='store_true')

    parser.add_argument('--bert_type', type=str, default='bert-base-cased')
    return parser

def save_model(model, path, name):
    torch.save(model.state_dict(), os.path.join(path, name))


def train(args):
    args.device = device = 'cuda'
    args.label2index = label2index = load_json('data/label_mapping.json')
    tokenizer = AutoTokenizer.from_pretrained(args.bert_type)
    writer = SummaryWriter()

    train_dataset = EDDataset('data/train.jsonlines', label2index, tokenizer, args)
    dev_dataset = EDDataset('data/dev.jsonlines', label2index, tokenizer, args)
    test_dataset = EDDataset('data/test.jsonlines', label2index, tokenizer, args)

    train_dl = DataLoader(train_dataset,
                          batch_size=args.batch_size,
                          num_workers=4,
                          shuffle=True,
                          collate_fn=EDDataset.pack)
    dev_dl = DataLoader(dev_dataset,
                        batch_size=args.batch_size,
                        num_workers=4,
                        shuffle=False,
                        collate_fn=EDDataset.pack)
    test_dl = DataLoader(test_dataset,
                         batch_size=args.batch_size,
                         num_workers=4,
                         shuffle=False,
                         collate_fn=EDDataset.pack)

    model = GRUModel(args).to(device)
    version = 'gru'
 
    params = [x for x in model.parameters() if x.requires_grad]
    weight = [1.0] + [10.0 for _ in range(1, len(label2index))]
    weight = torch.cuda.FloatTensor(weight)
    ce = CrossEntropyLoss(weight=weight, ignore_index=-100)
    optimizer = torch.optim.Adam(params, lr=args.lr)

    global_iter = 0
    best_dev = {'p': 0.0, 'r': 0.0, 'f': 0.0}
    best_test = {'p': 0.0, 'r': 0.0, 'f': 0.0}

    for epoch in range(args.epoch):
        model.train()
        bar = tqdm.tqdm(train_dl, desc='Training', total=len(train_dl))
        for batch in bar:
            global_iter += 1
            logits, preds = model(batch)
            # print('target', batch['target'].shape)

            loss = ce(logits, batch['target'].to(device))

            if global_iter % 10 == 0:
                l = loss.detach().cpu().numpy()
                writer.add_scalar('Loss/training', l, global_iter)
                bar.set_description(f'Training: Loss={l:.4f}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluation
        dev_perf = evaluate(model, dev_dl, writer, args, 'Dev', global_iter)
        # test_perf = evaluate(model, test_dl, writer, args, 'Test', global_iter)
        if dev_perf['f'] > best_dev['f']:
            best_dev = dev_perf
            # best_test = test_perf
            print('New best @ {}'.format(epoch))
            save_model(model, 'checkpoints', f'{version}_best.pth')
    
    model.load_state_dict(torch.load(f'checkpoints/{version}_best.pth'))
    evaluate(model, test_dl, writer, args, 'Test', 0)



def metrics(all_golds, all_preds, labels):
    p = precision_score(all_golds, all_preds, labels=labels, zero_division=0, average='micro')
    r = recall_score(all_golds, all_preds, labels=labels, zero_division=0, average='micro')
    f = f1_score(all_golds, all_preds, labels=labels, zero_division=0, average='micro')
    return {'p': p * 100, 'r': r * 100, 'f': f * 100}


def evaluate(model, dl, writer, args, msg='Test', global_iter=0):
    model.eval()
    all_golds = []
    all_preds = []

    labels = list(range(1, len(args.label2index)))

    for batch in tqdm.tqdm(dl, desc=msg):
        golds = batch['target'].numpy().tolist()
        all_golds += golds

        logits, preds = model(batch)
        preds = preds.cpu().numpy().tolist()
        all_preds += preds

    perfs = metrics(all_golds, all_preds, labels)
    writer.add_scalars('metric', perfs, global_iter)
    print('{}: {:.2f} {:.2f} {:.2f} '.format(msg,
                                             perfs['p'],
                                             perfs['r'],
                                             perfs['f'],
                                             ))
    return perfs


if __name__ == '__main__':
    args = arugment_parser().parse_args()
    train(args)
