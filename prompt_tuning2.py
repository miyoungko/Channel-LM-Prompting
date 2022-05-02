# import os
# from pathlib import Path

from pandas.core.base import NoNewAttributesMixin
from transformers import T5ForConditionalGeneration,T5Config
import torch
import torch.nn as nn
from soft_embedding import SoftEmbedding


class PromptTuningLM(T5ForConditionalGeneration):
    def __init__(self, 
                config,
                pretrained_model_name_or_path,
                n_tokens: int = None,
                initialize_from_vocab: bool = True,
                random_range: float = 0.5,):

        print(config)
        super(PromptTuningLM, self).__init__(config)
        print(pretrained_model_name_or_path)
        self.model = T5ForConditionalGeneration.from_pretrained(pretrained_model_name_or_path)
        self.n_tokens = n_tokens
        # Make sure to freeze Tranformers model
        for param in self.model.parameters():
            param.requires_grad = False
        

        self.s_wte = SoftEmbedding(self.model.get_input_embeddings(),
                        n_tokens=n_tokens,
                        initialize_from_vocab=initialize_from_vocab)
        self.model.set_input_embeddings(self.s_wte)

    
    def _cat_learned_embedding_to_input(self, input_ids):
        inputs_embeds = self.wte(input_ids)

        if len(list(inputs_embeds.shape)) == 2:
            inputs_embeds = inputs_embeds.unsqueeze(0)

        # [batch_size, n_tokens, n_embd]
        learned_embeds = self.soft_prompt.weight.repeat(inputs_embeds.size(0), 1, 1)

        inputs_embeds = torch.cat([learned_embeds, inputs_embeds], dim=1)

        return inputs_embeds

    def _extend_labels(self, labels, ignore_index=-100):
        if len(list(labels.shape)) == 1:
            labels = labels.unsqueeze(0)

        n_batches = labels.shape[0]
        return torch.cat(
            [
                torch.full((n_batches, self.n_tokens), ignore_index).to('cuda'),
                labels,
            ],
            dim=1,
        )

    def _extend_attention_mask(self, attention_mask):

        if len(list(attention_mask.shape)) == 1:
            attention_mask = attention_mask.unsqueeze(0)

        n_batches = attention_mask.shape[0]
        return torch.cat(
            [torch.full((n_batches, self.n_tokens), 1).to('cuda'), attention_mask],
            dim=1,
        )
    
    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        labels=None,
        past_key_values=None,
        head_mask=None,
        encoder_outputs=None,
        decoder_head_mask=None, 
        cross_attn_head_mask=None, 
        decoder_inputs_embeds=None, 
        use_cache=None, 
        output_attentions=None,
        output_hidden_states=None, 
        return_dict=None,
    ):

        if input_ids is None:
            # print("generation")
            # print(encoder_outputs)
            return super().forward(
                input_ids=input_ids, 
                decoder_input_ids = decoder_input_ids,
                encoder_outputs=encoder_outputs,
                past_key_values = past_key_values,
                attention_mask=attention_mask,
                labels=labels,
            )
        # print("training!")
        # print(self.input_ids)
        # print(input_ids, attention_mask,labels)
        if input_ids is not None:
            # inputs_embeds = self._cat_learned_embedding_to_input(input_ids).to('cuda')
            input_ids = torch.cat([torch.full((input_ids.shape[0], 10), 0).to('cuda'), input_ids], 1).to('cuda')

        if labels is not None:
            labels = self._extend_labels(labels).to('cuda')

        if attention_mask is not None:
            attention_mask = self._extend_attention_mask(attention_mask).to('cuda')
        # if self.input_ids is not None:


        # Drop most of the args for now
        # print(attention_mask.shape)
        return super().forward(
            # inputs_embeds=inputs_embeds, 
            attention_mask=attention_mask,
            # decoder_input_ids=decoder_input_ids,
            # decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            return_dict=return_dict,
            input_ids=input_ids
        )
    def generate(
        self,
        input_ids=None,
        attention_mask=None,
        use_cache=None,
        decoder_attention_mask=None,
        max_length=None,
        num_beams=None,
        early_stopping=None,
        min_length= None,
        do_sample= None,
        temperature= None,
        top_k= None,
        top_p= None,
        repetition_penalty= None,
        bad_words_ids= None,
        bos_token_id= None,
        pad_token_id= None,
        eos_token_id= None,
        length_penalty= None,
        no_repeat_ngram_size= None,
        encoder_no_repeat_ngram_size= None,
        num_return_sequences= None,
        max_time= None,
        max_new_tokens= None,
        decoder_start_token_id= None,
        num_beam_groups= None,
        diversity_penalty= None,
        prefix_allowed_tokens_fn= None,
        output_attentions=  None,
        output_hidden_states= None,
        output_scores= None,
        return_dict_in_generate= None,
        forced_bos_token_id= None,
        forced_eos_token_id= None,
        remove_invalid_values= None,
        synced_gpus= None,
        **model_kwargs,
    ):
        # print("intput_ids",input_ids)
        # self.generate_flag=1
        # self.input_ids = input_ids
        # self.attention_mask = attention_mask
        # print(super().get_input_embeddings(), "embedding")

        if input_ids is not None:
            print("input)ids", input_ids, input_ids.shape)
            input_ids = torch.cat([torch.full((input_ids.shape[0], 10), 0).to('cuda'), input_ids], 1).to('cuda')
        if attention_mask is not None:
            attention_mask = self._extend_attention_mask(attention_mask).to('cuda')

        return super().generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
            decoder_attention_mask=decoder_attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=early_stopping,
            min_length=min_length,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            bad_words_ids=bad_words_ids,
            bos_token_id=bos_token_id,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size,
            num_return_sequences=num_return_sequences,
            max_time=max_time,
            max_new_tokens=max_new_tokens,
            decoder_start_token_id=decoder_start_token_id,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_scores=output_scores,
            return_dict_in_generate=return_dict_in_generate,
            forced_bos_token_id=forced_bos_token_id,
            forced_eos_token_id=forced_eos_token_id,
            remove_invalid_values=remove_invalid_values,
            synced_gpus=synced_gpus,
            **model_kwargs,
        )

