from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments, AutoConfig, PreTrainedModel
import torch
from torch import nn
import torch.nn.functional as F

class InfoNCEDualEncoder(PreTrainedModel):
    def __init__(self, 
                 config, 
                 query_model_name="/home/nlp/achimoa/workspace/ModernBERT/hf/HebrewModernBERT/ModernBERT-Hebrew-base_20250522_1841", 
                 doc_model_name=None, 
                 pooling='cls'):
        super().__init__(config)
        self.query_encoder = AutoModel.from_pretrained(query_model_name)
        if doc_model_name:
            self.doc_encoder = AutoModel.from_pretrained(doc_model_name)
        else:
            self.doc_encoder = AutoModel.from_pretrained(query_model_name)
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


