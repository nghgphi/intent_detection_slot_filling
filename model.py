from module import IntentClassifier, SlotClassifier
import torch
import torch.nn as nn
from torchcrf import CRF 
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaPreTrainedModel

class IDSF_PhoBERT(RobertaPreTrainedModel):
    def __init__(self, config, args, intent_labels, slot_labels):
        super(IDSF_PhoBERT, self).__init__(config)
        self.args = args
        self.num_intent_labels = len(intent_labels)
        self.num_slot_labels = len(slot_labels)
        self.roberta = RobertaModel(config)

        self.intent_classifier = IntentClassifier(config.hidden_size, self.num_intent_labels, args.dropout_rate) 
        self.slot_classifier = SlotClassifier(
            input_dim= config.hidden_size, 
            num_intent_labels= self.num_intent_labels,
            num_slot_labels= self.num_slot_labels,
            use_intent_context_concat= self.args.use_intent_context_concat,
            use_intent_context_attn= self.args.use_intent_context_attn,
            max_seq_len= self.args.max_seq_len,
            attention_embedding_size= self.args.attention_embedding_size,
            d_r = self.args.d_r
        )

        if args.use_crf:
            self.crf = CRF(num_tags= self.num_slot_labels, batch_first=True)
    def forward(self, input_ids, attention_mask, token_type_ids, intent_label_ids, slot_label_ids):
        output = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = output[0]
        pooled_output = output[1]
        
        intent_logits = self.intent_classifier(pooled_output)
        
        if not self.args.use_attention_mask:
            tmp_attention_mask = None
        else:
            tmp_attention_mask = attention_mask
        
        if self.args.embedding_type == 'hard':
            hard_intent_logits = torch.zeros(intent_logits.shape)
            
            for i, sample in enumerate(intent_logits):
                max_idx = torch.argmax(sample)
                hard_intent_logits[i][max_idx] = 1
            
            slot_logits = self.slot_classifier(sequence_output, hard_intent_logits, tmp_attention_mask)
        else:
            slot_logits = self.slot_classifier(sequence_output, intent_logits, tmp_attention_mask)
        
        #_______________________________________
        total_loss = 0
        
        if intent_label_ids is not None:
            if self.num_intent_labels == 1:
                intent_loss_func = nn.MSELoss()
                intent_loss = intent_loss_func(intent_logits.view(-1), intent_label_ids.view(-1))
            else:
                intent_loss_func = nn.CrossEntropyLoss()
                intent_loss = intent_loss_func(intent_logits.view(-1, self.num_intent_labels), intent_label_ids.view(-1))
            total_loss += self.args.intent_loss_coef * intent_loss
        
        if slot_label_ids is not None:
            if self.args.use_crf:
                slot_loss = self.crf(slot_logits, slot_label_ids, mask= attention_mask.byte(), reduction= 'mean')
                slot_loss = -1 * slot_loss
            else:
                slot_loss_func = nn.CrossEntropyLoss(ignore_index=self.args.ignore_index)
                
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = slot_logits.view(-1, self.num_slot_labels)[active_loss]
                    active_labels = slot_label_ids.view(-1)[active_loss]
                    
                    slot_loss = slot_loss_func(active_logits, active_labels)
                else:
                    slot_loss = slot_loss_func(slot_logits.view(-1, self.num_slot_labels), slot_label_ids.view(-1))
                    
            total_loss += (1 - self.args.intent_loss_coef) * slot_loss  
        
        outputs = ((intent_logits, slot_logits), ) + output[2:]
        outputs = (total_loss,) + outputs 
        return outputs
        
        