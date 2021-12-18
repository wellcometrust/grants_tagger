from transformers import AutoModel
import torch


class MultiLabelAttention(torch.nn.Module):
    def __init__(self, D_in, num_labels):
        super().__init__()
        self.A = torch.nn.Parameter(torch.empty(D_in, num_labels))
        torch.nn.init.uniform_(self.A, -0.1, 0.1)

    def forward(self, x):
        attention_weights = torch.nn.functional.softmax(
            torch.tanh(torch.matmul(x, self.A)), dim=1
        )
        return torch.matmul(torch.transpose(attention_weights, 2, 1), x)


class BertMesh(torch.nn.Module):
    def __init__(
        self,
        pretrained_model,
        num_labels,
        hidden_size=512,
        dropout=0,
        multilabel_attention=False,
    ):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.num_labels = num_labels
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.multilabel_attention = multilabel_attention

        self.bert = AutoModel.from_pretrained(pretrained_model)  # 768
        self.multilabel_attention_layer = MultiLabelAttention(
            768, num_labels
        )  # num_labels, 768
        self.linear_1 = torch.nn.Linear(768, hidden_size)  # num_labels, 512
        self.linear_2 = torch.nn.Linear(hidden_size, 1)  # num_labels, 1
        self.linear_out = torch.nn.Linear(hidden_size, num_labels)
        self.dropout_layer = torch.nn.Dropout(self.dropout)

    def forward(self, inputs):
        if self.multilabel_attention:
            hidden_states = self.bert(input_ids=inputs)[0]
            attention_outs = self.multilabel_attention_layer(hidden_states)
            outs = torch.nn.functional.relu(self.linear_1(attention_outs))
            outs = self.dropout_layer(outs)
            outs = torch.sigmoid(self.linear_2(outs))
            outs = torch.flatten(outs, start_dim=1)
        else:
            cls = self.bert(input_ids=inputs)[1]
            outs = torch.nn.functional.relu(self.linear_1(cls))
            outs = self.dropout_layer(outs)
            outs = torch.sigmoid(self.linear_out(outs))
        return outs
