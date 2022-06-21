import torch

from torch.nn import *
from transformers import AutoModel

class BertEmbedding(Module):

    def __init__(self, args):
        super(BertEmbedding, self).__init__()
        self.device = args.device
        print('Init Bert')
        self.bert = AutoModel.from_pretrained(args.bert_type)

        if not args.update_bert:
            for params in self.bert.parameters():
                params.requires_grad = False
        self.output_size = 8 * self.bert.config.hidden_size

    def forward(self, inputs):

        BL = torch.max(inputs['bert_length'])

        indices = inputs['indices'][:, :BL].to(self.device)
        attention_mask = inputs['attention_mask'][:, :BL].to(self.device)

        bert_outputs = self.bert(indices,
                                 attention_mask=attention_mask,
                                 output_hidden_states=True)

        bert_x = torch.concat(bert_outputs['hidden_states'][-8:], dim=-1)  # B x L x D
        # print('Bert x', tuple(bert_x.shape))

        all_embeddings = []

        for sid, sentence_spans in enumerate(inputs['word_spans']):
            for start, end in sentence_spans:
                emb = torch.mean(bert_x[sid][start:end], dim=-2)
                # print('emb', tuple(emb.shape))
                all_embeddings.append(emb)
        all_embeddings = torch.stack(all_embeddings)

        return all_embeddings


class MLPModel(Module):

    def __init__(self, args):
        super(MLPModel, self).__init__()
        self.device = args.device
        self.c = len(args.label2index)
        self.embeddings = BertEmbedding(args)
        self.fc = Sequential(
            Linear(self.embeddings.output_size, 1024),
            Dropout(),
            Tanh(),
            Linear(1024, self.c)
        )

    def forward(self, inputs):
        embeddings = self.embeddings(inputs)
        logits = self.fc(embeddings)
        preds = torch.argmax(logits, dim=-1)

        # print('logits', tuple(logits.shape))
        # print('preds', tuple(preds.shape))
        return logits, preds


class CNNModel(Module):

    def __init__(self, args):
        super(CNNModel, self).__init__()
        self.device = args.device
        self.c = len(args.label2index)
        self.embeddings = BertEmbedding(args)

        self.cnn = Sequential(
            Conv1d(8*768,2048,3, padding='same'),
            ReLU(),
            Conv1d(2048,1024,5, padding='same'),
            ReLU(),
        )
        #torch.nn.funcional.F.maxpool1d
        #resolve length
        
        self.fc = Sequential(
            Linear(1024, 1024),
            Dropout(),
            Tanh(),
            Linear(1024, self.c)
        )

    def forward(self, inputs):
        embeddings = self.embeddings(inputs)
        x = torch.transpose(embeddings,0,1) #(B,L,M*D) -> transpose -> (B,M*D,L)
        x = self.cnn(x)
        x = torch.transpose(x,0,1)
        logits = self.fc(x)
        preds = torch.argmax(logits, dim=-1)

        return logits, preds

#B: batch_size
#L: sentence lenght
#M: 8
#D: 768

class LSTMModel(Module):

    def __init__(self, args):
        super(LSTMModel, self).__init__()
        self.device = args.device
        self.c = len(args.label2index)
        self.embeddings = BertEmbedding(args)

        self.lstm = torch.nn.LSTM(8*768, 1024)

        #torch.nn.funcional.F.maxpool1d
        #resolve length
        
        self.fc = Sequential(
            Linear(1024, 1024),
            Dropout(),
            Tanh(),
            Linear(1024, self.c)
        )

    def forward(self, inputs):
        embeddings = self.embeddings(inputs)
        x = embeddings.view(-1,1,8*768) #(B_L,1,M*D)
        out, hidden = self.lstm(x)
        out = out.view(-1, 1024)
        # x = torch.transpose(x,0,1)
        # print(out.shape)
        logits = self.fc(out)
        preds = torch.argmax(logits, dim=-1)

        return logits, preds



class GRUModel(Module):

    def __init__(self, args):
        super(GRUModel, self).__init__()
        self.device = args.device
        self.c = len(args.label2index)
        self.embeddings = BertEmbedding(args)

        self.gru = torch.nn.GRU(8*768, 1024)
        
        self.fc = Sequential(
            Linear(1024, 1024),
            Dropout(),
            Tanh(),
            Linear(1024, self.c)
        )

    def forward(self, inputs):
        embeddings = self.embeddings(inputs)
        x = embeddings.view(-1,1,8*768) #(B_L,1,M*D)
        out, hidden = self.gru(x)
        # print(out.shape)
        out = out.view(-1, 1024)
        logits = self.fc(out)
        preds = torch.argmax(logits, dim=-1)

        return logits, preds

if __name__ == '__main__':

    # args = arugment_parser().parse_args()

    input = torch.randn(20,1, 8*768)
    model = torch.nn.LSTM(8*768, 1024)
    out, hidden = model(input)
    print(out.shape)
    pass