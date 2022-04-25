from transformers import BertModel, BertTokenizer
from torch.utils.data import Dataset, DataLoader
# from datasets import load_dataset, DatasetDict
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import CrossEntropyLoss
import torch
import numpy as np

class AttentionLayer(nn.Module):
    def __init__(self, dimensions):
        super(AttentionLayer, self).__init__()
        self.dimensions = dimensions
        self.linear_out = nn.Linear(dimensions * 2, dimensions, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
    def forward(self, query, context, attention_mask):
        batch_size, output_len, hidden_size = query.size()

        attention_scores = torch.bmm(query, context.transpose(1, 2).contiguous())

        if attention_mask is not None:
            attention_mask = torch.unsqueeze(attention_mask, 2)
            attention_scores.masked_fill_(attention_mask == 0, -np.inf)
        
        attention_weights = self.softmax(attention_scores)

        mix = torch.bmm(attention_weights, query)
        combined = torch.cat((mix,query), dim=2)
        
        output = self.linear_out(combined)
        output = self.tanh(output)

        return output, attention_weights

class IntentClassifier(nn.Module):
    def __init__(self, input_dim, num_intent_labels, dropout_rate=0.0):
        super(IntentClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_intent_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)

class SlotClassifier(nn.Module):
    def __init__(self, 
        input_dim, 
        num_intent_labels, 
        num_slot_labels, 
        use_intent_context_concat=False, 
        use_intent_context_attn=False, 
        max_seq_len=50, 
        attention_embedding_size=200, 
        d_r=0.0
    ):
        super(SlotClassifier, self).__init__()
        self.use_intent_context_attn = use_intent_context_attn
        self.use_intent_context_concat = use_intent_context_concat
        self.max_seq_len = max_seq_len
        self.num_intent_labels = num_intent_labels
        self.num_slot_labels = num_slot_labels
        self.attention_embedding_size = attention_embedding_size

        output_dim = self.attention_embedding_size
        if self.use_intent_context_concat:
            output_dim = self.attention_embedding_size
            self.linear_out = nn.Linear(2 * attention_embedding_size, attention_embedding_size)
        elif self.use_intent_context_attn:
            output_dim = self.attention_embedding_size
            self.attention = AttentionLayer(attention_embedding_size)

        self.linear_slot = nn.Linear(input_dim, self.attention_embedding_size, bias=False)

        if self.use_intent_context_attn or self.use_intent_context_concat:
            self.linear_intent_context = nn.Linear(self.num_intent_labels, self.attention_embedding_size, bias=False) 
            self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(d_r)
        self.linear = nn.Linear(output_dim, num_slot_labels)

    def forward(self, x, intent_context, attention_mask):
        x = self.linear_slot(x)

        if self.use_intent_context_concat:
            intent_context = self.softmax(intent_context)
            intent_context = self.linear_intent_context(intent_context) 
            intent_context = torch.unsqueeze(intent_context, 1)
            intent_context = intent_context.expand(-1, self.max_seq_len, -1)
            x = torch.cat((x, intent_context), 2)
            x = self.linear_out(x) 
        
        elif self.use_intent_context_attn:
            intent_context = self.softmax(intent_context)
            intent_context = self.linear_intent_context(intent_context)
            intent_context = torch.unsqueeze(intent_context, 1)
            output, weights = self.attention(x, intent_context, attention_mask)
            x = output
        x = self.dropout(x)
        return self.linear(x)
        
        

    


# from vncorenlp import VnCoreNLP
# annotator = VnCoreNLP("VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx2g') 

# # Input 
# text = "Ông Nguyễn Khắc Chúc  đang làm việc tại Đại học Quốc gia Hà Nội. Bà Lan, vợ ông Chúc, cũng làm việc tại đây."

# # To perform word segmentation, POS tagging, NER and then dependency parsing
# annotated_text = annotator.annotate(text)   

# # To perform word segmentation only
# word_segmented_text = annotator.tokenize(text)
# print(word_segmented_text)