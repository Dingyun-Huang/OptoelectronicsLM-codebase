from transformers import RobertaModel, BertModel, AlbertModel
import torch
import torch.nn as nn


class Pooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        # self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        pooled_output = self.dense(hidden_states)
        # pooled_output = self.dropout(pooled_output)
        # print(pooled_output.shape)
        # pooled_output = self.norm(pooled_output)
        # print(pooled_output.shape)
        pooled_output = self.activation(pooled_output)
        # print('-'*128)
        return pooled_output


class SentenceRoberta(nn.Module):

    def __init__(self, base_model: str, pooling_mode: str):
        super(SentenceRoberta, self).__init__()

        self.model = RobertaModel.from_pretrained(base_model, output_hidden_states=True, output_attentions=True, add_pooling_layer=False)
        self.pooler = Pooler(self.model.config)
        # self.pooler = RobertaOutput(self.model.config)

        if pooling_mode in {"mean", "cls"}:
            self.pooling_mode = pooling_mode
        else:
            raise ValueError(f"Invalid pooling mode: {pooling_mode}")

    def forward(self, input_ids=None, attention_mask=None):
        #Extract outputs from the body
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        #Add custom layers
        if self.pooling_mode == "mean":
            # mean pooling
            # print(outputs.pooler_output.shape)
            pooled_outputs = outputs.hidden_states[0]
            pooled_outputs = self.pooler(pooled_outputs)
            sentence_embedding = pooled_outputs.mean(dim=1)
            # assert sentence_embedding.shape == (input_ids.shape[0], 768)
        else:
            # cls token pooling
            pooled_outputs = outputs.hidden_states[0]
            pooled_outputs = self.pooler(pooled_outputs)
            sentence_embedding = pooled_outputs[:, 0, :]
            # assert sentence_embedding.shape == (input_ids.shape[0], 768)

        return sentence_embedding


class SentenceBert(nn.Module):

    def __init__(self, base_model: str, pooling_mode: str):
        super(SentenceBert, self).__init__()

        self.model = BertModel.from_pretrained(base_model, output_hidden_states=True, output_attentions=True, add_pooling_layer=False)
        self.pooler = Pooler(self.model.config)
        # self.pooler = RobertaOutput(self.model.config)

        if pooling_mode in {"mean", "cls"}:
            self.pooling_mode = pooling_mode
        else:
            raise ValueError(f"Invalid pooling mode: {pooling_mode}")

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
        #Extract outputs from the body
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        #Add custom layers
        if self.pooling_mode == "mean":
            # mean pooling
            # print(outputs.pooler_output.shape)
            pooled_outputs = outputs.hidden_states[0]
            pooled_outputs = self.pooler(pooled_outputs)
            sentence_embedding = pooled_outputs.mean(dim=1)
            # assert sentence_embedding.shape == (input_ids.shape[0], 768)
        else:
            # cls token pooling
            pooled_outputs = outputs.hidden_states[0]
            pooled_outputs = self.pooler(pooled_outputs)
            sentence_embedding = pooled_outputs[:, 0, :]
            # assert sentence_embedding.shape == (input_ids.shape[0], 768)

        return sentence_embedding


class SentenceAlbert(nn.Module):

    def __init__(self, base_model: str, pooling_mode: str):
        super(SentenceAlbert, self).__init__()

        self.model = AlbertModel.from_pretrained(base_model, output_hidden_states=True, output_attentions=True, add_pooling_layer=False)
        self.pooler = Pooler(self.model.config)
        # self.pooler = RobertaOutput(self.model.config)

        if pooling_mode in {"mean", "cls"}:
            self.pooling_mode = pooling_mode
        else:
            raise ValueError(f"Invalid pooling mode: {pooling_mode}")

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
        #Extract outputs from the body
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        #Add custom layers
        if self.pooling_mode == "mean":
            # mean pooling
            # print(outputs.pooler_output.shape)
            pooled_outputs = outputs.hidden_states[0]
            pooled_outputs = self.pooler(pooled_outputs)
            sentence_embedding = pooled_outputs.mean(dim=1)
            # assert sentence_embedding.shape == (input_ids.shape[0], 768)
        else:
            # cls token pooling
            pooled_outputs = outputs.hidden_states[0]
            pooled_outputs = self.pooler(pooled_outputs)
            sentence_embedding = pooled_outputs[:, 0, :]
            # assert sentence_embedding.shape == (input_ids.shape[0], 768)

        return sentence_embedding


class GteLoss(nn.Module):
    def __init__(self):
        super(GteLoss, self).__init__()

    def forward(self, anchor_embeddings, positive_embeddings, temperature=torch.tensor(0.05)):
        # Calculate the loss
        # embeddings shape = (batch_size, 768)
        # temperature is a scalar
        # print(anchor_embeddings.shape)
        loss = torch.tensor(0.0, device=anchor_embeddings.device)
        for i in range(anchor_embeddings.shape[0]):
            partition = self.partition_function(anchor_embeddings[i, :], positive_embeddings, temperature) \
                        + self.partition_function(anchor_embeddings[i, :], anchor_embeddings, temperature) - torch.exp(1 / temperature) \
                        + self.partition_function(anchor_embeddings, positive_embeddings[i, :], temperature) \
                        + self.partition_function(positive_embeddings, positive_embeddings[i, :], temperature) - torch.exp(1 / temperature)
            loss -= torch.log(self.partition_function(anchor_embeddings[i, :], positive_embeddings[i, :], temperature) / partition)
        loss = loss / anchor_embeddings.shape[0]
        return loss
        

    def partition_function(self, embeddingA, embeddingB, temperature):
        # Calculate the partition function
        # embeddings shape = (batch_size, 768)
        cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
        exponents = cos(embeddingA, embeddingB) / temperature
        # print(exponents)
        partition = torch.exp(exponents).sum()
        # print(partition)
        return partition


class InfoNCELoss(nn.Module):
    def __init__(self):
        super(InfoNCELoss, self).__init__()

    def forward(self, anchor_embeddings, positive_embeddings, temperature=torch.tensor(0.05)):
        # Calculate the loss
        # embeddings shape = (batch_size, 768)
        # temperature is a scalar
        # print(anchor_embeddings.shape)
        loss = torch.tensor(0.0, device=anchor_embeddings.device)
        for i in range(anchor_embeddings.shape[0]):
            partition = self.partition_function(anchor_embeddings[i, :], positive_embeddings, temperature)
            loss -= torch.log(self.partition_function(anchor_embeddings[i, :], positive_embeddings[i, :], temperature) / partition)
        loss = loss / anchor_embeddings.shape[0]
        return loss
        

    def partition_function(self, embeddingA, embeddingB, temperature):
        # Calculate the partition function
        # embeddings shape = (batch_size, 768)
        cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
        exponents = cos(embeddingA, embeddingB) / temperature
        # print(exponents)
        partition = torch.exp(exponents).sum()
        # print(partition)
        return partition
