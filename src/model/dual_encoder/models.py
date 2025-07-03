from transformers import PretrainedConfig, PreTrainedModel, AutoModel
import torch.nn as nn
import torch
import torch.nn.functional as F

class InfoNCEDualEncoderConfig(PretrainedConfig):
    model_type = "info_nce_dual_encoder"

    def __init__(self, query_model_name=None, doc_model_name=None, 
                 query_tokenizer_path=None, doc_tokenizer_path=None,
                 pooling="cls", temperature=0.05, **kwargs):
        super().__init__(**kwargs)
        self.query_model_name = query_model_name
        self.doc_model_name = doc_model_name
        self.query_tokenizer_path = query_tokenizer_path
        self.doc_tokenizer_path = doc_tokenizer_path
        self.pooling = pooling
        self.temperature = temperature


class InfoNCEDualEncoder(PreTrainedModel):
    config_class = InfoNCEDualEncoderConfig

    def __init__(self, config):
        super().__init__(config)
        self.query_encoder = AutoModel.from_pretrained(config.query_model_name)
        self.doc_encoder = AutoModel.from_pretrained(config.doc_model_name or config.query_model_name)
        self.pooling = config.pooling
        self.temperature = config.temperature

    def _pool(self, output, attention_mask):
        if self.pooling == 'mean':
            token_embeddings = output.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
            return (token_embeddings * input_mask_expanded).sum(1) / input_mask_expanded.sum(1)
        else:
            return output.last_hidden_state[:, 0]

    def forward(self, query_input_ids, query_attention_mask, doc_input_ids, doc_attention_mask, labels=None):
        q_out = self.query_encoder(input_ids=query_input_ids, attention_mask=query_attention_mask)
        d_out = self.doc_encoder(input_ids=doc_input_ids, attention_mask=doc_attention_mask)

        q_emb = F.normalize(self._pool(q_out, query_attention_mask), dim=-1)
        d_emb = F.normalize(self._pool(d_out, doc_attention_mask), dim=-1)

        logits = torch.matmul(q_emb, d_emb.T) / self.temperature
        targets = torch.arange(logits.size(0), device=logits.device)
        loss = F.cross_entropy(logits, targets)

        return {"loss": loss, "logits": logits}

    def encode(self, encoder, input_ids, attention_mask):
        output = encoder(input_ids=input_ids, attention_mask=attention_mask)

        if self.pooling == 'cls':
            emb = output.last_hidden_state[:, 0]
        elif self.pooling == 'mean':
            mask = attention_mask.unsqueeze(-1).expand(output.last_hidden_state.size())
            sum_emb = torch.sum(output.last_hidden_state * mask, dim=1)
            sum_mask = torch.clamp(mask.sum(1), min=1e-9)
            emb = sum_emb / sum_mask
        else:
            raise ValueError("Unknown pooling type")

        # Normalize embeddings for cosine similarity
        emb = torch.nn.functional.normalize(emb, p=2, dim=-1)

        return emb



class InfoNCEDualEncoder2(nn.Module):
    def __init__(self, 
                 query_model_name,
                 doc_model_name=None, 
                 pooling='cls'):
        super().__init__()
        self.query_encoder = AutoModel.from_pretrained(query_model_name)
        self.doc_encoder = AutoModel.from_pretrained(doc_model_name or query_model_name)
        self.pooling = pooling

    def encode(self, encoder, input_ids, attention_mask):
        output = encoder(input_ids=input_ids, attention_mask=attention_mask)
        # [CLS] pooling or mean pooling
        if self.pooling == 'cls':
            return output.last_hidden_state[:, 0]  # [batch, hidden]
        elif self.pooling == 'mean':
            mask = attention_mask.unsqueeze(-1).expand(output.last_hidden_state.size())
            sum_emb = torch.sum(output.last_hidden_state * mask, 1)
            sum_mask = torch.clamp(mask.sum(1), min=1e-9)
            return sum_emb / sum_mask
        else:
            raise ValueError("Unknown pooling type")

    def forward(
        self,
        q_input_ids,
        q_attention_mask,
        d_input_ids,
        d_attention_mask,
        labels=None  # not used
    ):
        # [batch, hidden]
        q_emb = self.encode(self.query_encoder, q_input_ids, q_attention_mask)
        d_emb = self.encode(self.doc_encoder, d_input_ids, d_attention_mask)

        # [batch, batch] similarity matrix
        sim_matrix = torch.matmul(q_emb, d_emb.T)  # dot product, or use F.cosine_similarity

        # InfoNCE loss
        targets = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
        loss = F.cross_entropy(sim_matrix, targets)

        return {"loss": loss, "logits": sim_matrix}


