from transformers import AutoModel, PreTrainedModel, BertConfig
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


class BertMesh(PreTrainedModel):
    config_class = BertConfig

    def __init__(
        self,
        config,
    ):
        super().__init__(config=config)
        self.config.auto_map = {"AutoModel": "model.BertMesh"}
        self.pretrained_model = self.config.pretrained_model
        self.num_labels = self.config.num_labels
        self.hidden_size = getattr(self.config, "hidden_size", 512)
        self.dropout = getattr(self.config, "dropout", 0.1)
        self.multilabel_attention = getattr(self.config, "multilabel_attention", False)
        self.id2label = self.config.id2label
        self.threshold = getattr(self.config, "threshold", 0.5)

        self.bert = AutoModel.from_pretrained(self.pretrained_model)  # 768
        self.multilabel_attention_layer = MultiLabelAttention(
            768, self.num_labels
        )  # num_labels, 768
        self.linear_1 = torch.nn.Linear(768, self.hidden_size)  # num_labels, 512
        self.linear_2 = torch.nn.Linear(self.hidden_size, 1)  # num_labels, 1
        self.linear_out = torch.nn.Linear(self.hidden_size, self.num_labels)
        self.dropout_layer = torch.nn.Dropout(self.dropout)

    def forward(self, input_ids, return_labels=False, **kwargs):
        if type(input_ids) is list:
            # coming from tokenizer
            input_ids = torch.tensor(input_ids)
        if self.multilabel_attention:
            hidden_states = self.bert(input_ids=input_ids)[0]
            attention_outs = self.multilabel_attention_layer(hidden_states)
            outs = torch.nn.functional.relu(self.linear_1(attention_outs))
            outs = self.dropout_layer(outs)
            outs = torch.sigmoid(self.linear_2(outs))
            outs = torch.flatten(outs, start_dim=1)
        else:
            cls = self.bert(input_ids=input_ids)[1]
            outs = torch.nn.functional.relu(self.linear_1(cls))
            outs = self.dropout_layer(outs)
            outs = torch.sigmoid(self.linear_out(outs))
        if return_labels:
            # TODO Vectorize
            outs = [
                [
                    self.id2label[label_id]
                    for label_id, label_prob in enumerate(out)
                    if label_prob > self.threshold
                ]
                for out in outs
            ]
        return outs
