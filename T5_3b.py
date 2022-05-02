
import pytorch_lightning as pl
from transformers import T5Tokenizer, T5ForConditionalGeneration, Adafactor, T5Config
from models.prompt_tuning import MyEmbedding
import csv 
from torch.utils.data import RandomSampler
from torch.utils.data import DataLoader, ConcatDataset
from Datasets import Pretrain
from evaluation import (
    ids_to_clean_text,
    calculate_em_multipleanswers,
    calculate_em_multipleanswers_overlap,
    calculate_accuracy_multipleanswers,
    calculate_rouge_multipleanswers,
    calculate_f1_scores,
    calculate_accuracy_scores,
    calculate_rouge_scores,
    calculate_em_scores,
)
import torch
from torch.optim import AdamW
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
import os
import functools

from models.Prefixtuning import T5ForConditionalGeneration as PrefixTransformer
from models.Kadapter import T5ForConditionalGeneration as T5_Kadapter
from models.Lora import T5ForConditionalGeneration as T5_Lora
from models.Residuals import T5ForConditionalGeneration as T5_Residuals
from models.Token_adapter import T5ForConditionalGeneration as T5_Token

def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


class T5_3b(pl.LightningModule):
    def __init__(self, args):
        super(T5_3b, self).__init__()
        self.args = args
        self.n_prefix = self.args.n_prefix
        self.tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
        self.total_loss = 0
        self.iteration = 0
        if args.method == 'prefixtuning':
            self.model = PrefixTransformer.from_pretrained(args.model_name_or_path)
        elif args.method == 'prompt_tune':
            task_tokens = ["<TASK{}>".format(str(i).zfill(2)) for i in range(self.n_prefix)]
            self.tokenizer.add_tokens(task_tokens, special_tokens=True)
            task_token_ids = self.tokenizer(" ".join(task_tokens), return_tensors="pt", add_special_tokens=False)["input_ids"]
            self.model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
            self.freeze_params(self.model)
            self.model.set_input_embeddings(MyEmbedding(self.model.shared, self.n_prefix))
            if self.args.adapter_checkpoint_path !="" and self.args.mode =='evaluate':
                param_dict = torch.load(self.args.adapter_checkpoint_path)
                for name, param in param_dict.items():
                    rsetattr(self.model, name,torch.nn.Parameter(param.to('cuda')))
        elif args.method == 'kadapter':
            self.model = T5_Kadapter.from_pretrained(args.model_name_or_path)
            self.freeze_params(self.model.get_encoder()) #Freezing the encoder
            # Unfreezing the parameters used for kadapters in encoder
            for name, param in self.model.named_parameters():
                if 'kadapter' in name:
                    param.requires_grad = True
        elif args.method == 'lora':
            self.model = T5_Lora.from_pretrained(args.model_name_or_path)
            self.freeze_params(self.model)
            for name, param in self.model.named_parameters():
                if 'lora' in name:
                    param.requires_grad = True
        elif args.method == 'residual':
            self.model = T5_Residuals.from_pretrained(args.model_name_or_path, scale = self.args.scale)
            if self.args.adapter_checkpoint_path !="" and self.args.mode =='evaluate':
                param_dict = torch.load(self.args.adapter_checkpoint_path)
                for name, param in param_dict.items():
                    rsetattr(self.model, name,torch.nn.Parameter(param.to('cuda')))
            self.freeze_params(self.model)
            for name, param in self.model.named_parameters():
                if 'residual' in name:
                    param.requires_grad = True
        elif args.method == 'token_adapter': 
            self.model = T5_Token.from_pretrained(args.model_name_or_path, scale = self.args.scale)
            if self.args.adapter_checkpoint_path !="" and self.args.mode =='evaluate':
                param_dict = torch.load(self.args.adapter_checkpoint_path)
                for name, param in param_dict.items():
                    rsetattr(self.model, name,torch.nn.Parameter(param.to('cuda')))
            self.freeze_params(self.model)
            for name, param in self.model.named_parameters():
                if 'residual' in name:
                    param.requires_grad = True
        else:
            self.model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
        #Freezing only encoder or the whole model
        if args.freeze_level==1: # Freeze encoder only
            self.freeze_params(self.model.get_encoder())
        elif args.freeze_level==2: # Freeze encoder and decoder
            self.freeze_params(self.model) 
        elif args.freeze_level==3:
            self.freeze_params(self.model)
            for name, param in self.model.named_parameters():
                if 'lm_head' in name:
                    param.requires_grad = True
        elif args.freeze_level==4:
            self.freeze_params(self.model)
            for name, param in self.model.named_parameters():
                if 'control_trans' in name or 'wte' in name:
                    param.requires_grad = True

    def freeze_params(self, model):
        for par in model.parameters():
            par.requires_grad = False
    
    def get_dataset(self, dataset, tokenizer, type_path, args):
        dataset = Pretrain(dataset=dataset, tokenizer=tokenizer, type_path=type_path, input_length=args.max_input_length, 
                        output_length=args.max_output_length, args=args)
        if type_path == 'validation':
            self.ids_to_answers = dataset.ids_to_answers
        return dataset

    
    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, lm_labels=None):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=lm_labels,
        )
    
    def _step(self, batch):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            lm_labels=lm_labels,
            decoder_attention_mask=batch['target_mask']
        )

        loss = outputs[0]
        return loss
    
    def _generative_step(self, batch, batch_idx):  
        if self.args.method == 'prefixtuning':
            past_key_values = self.model.get_prompt_p5(bsz=len(batch["source_ids"]))
            source = ids_to_clean_text(self.tokenizer, batch["source_ids"])
            generated_ids = self.model.generate(
                batch["source_ids"],
                attention_mask=batch["source_mask"],
                use_cache=True,
                past_key_values= past_key_values,
                decoder_attention_mask=batch['target_mask'],
                max_length=self.args.max_output_length,
                num_beams=2,
                early_stopping=True
            )
        else: 
            source = ids_to_clean_text(self.tokenizer, batch["source_ids"])
            generated_ids = self.model.generate(
                batch["source_ids"],
                attention_mask=batch["source_mask"],
                use_cache=True,
                decoder_attention_mask=batch['target_mask'],
                max_length=self.args.max_output_length,
                num_beams=2,
                early_stopping=True
            )
        preds = ids_to_clean_text(self.tokenizer, generated_ids)
        targets = ids_to_clean_text(self.tokenizer, batch["target_ids"])
        data_label = batch["data_label"]

        loss = self._step(batch)
        self.log("val_loss", loss)
        
        print("preds", preds)
        print("targets", targets)

        em_score = 0
        accuracy = 0
        rouge_score = 0
        f1_score = 0

        # if self.args.mode == 'zerotune':
        if 'multi_news' in data_label or 'cnn_dailymail' in data_label or 'gigaword' in data_label or 'samsum' in data_label or 'xsum' in data_label or 'common_gen' in data_label or 'wiki_bio' in data_label:
            rouge_score = calculate_rouge_scores(preds, targets)
        elif ('imdb' in data_label or 'amazon_polarity' in data_label or 'rotten_tomatoes' in data_label or 'yelp_review_full' in data_label or 'glue/mrpc' in data_label or 'glue/qqp' in data_label or 'paws' in data_label
            or "ag_news" in data_label or "trec" in data_label or "dbpedia_14" in data_label or "cos_e" in data_label or "dream" in data_label or "qasc" in data_label or "quarel" in data_label or "sciq" in data_label 
            or "quail" in data_label or "quartz" in data_label or "wiqa" in data_label or "wiki_hop" in data_label or "social_i_qa" in data_label or "cosmos_qa" in data_label) or self.args.dataset == 'glue':
            accuracy = calculate_accuracy_scores(preds, targets)
        elif 'adversarial_qa/dbidaf' in data_label or 'adversarial_qa/dbert' in data_label or 'adversarial_qa/droberta' in data_label or 'quoref' in data_label or self.args.dataset_config == 'hotpotqa' or 'wiki_qa' in data_label:
            em_score = calculate_em_scores(preds, targets)
        elif self.args.dataset == 'lama' or self.args.dataset == 'TriviaQA':
            f1_score = calculate_f1_scores(preds, targets)
            em_score = calculate_em_scores(preds, targets)
        elif self.args.dataset == 'super_glue' or self.args.dataset == 'anli':
            accuracy = calculate_accuracy_scores(preds, targets)
        else:
            raise NameError('Select the correct Dataset!')

        em_score = torch.tensor(em_score,dtype=torch.float32)
        accuracy = torch.tensor(accuracy,dtype=torch.float32)
        rouge_score = torch.tensor(rouge_score, dtype=torch.float32)
        f1_score = torch.tensor(f1_score, dtype=torch.float32)
        # if self.args.mode == 'zerotune':
        if self.args.output_log != None:
            f = open(self.args.output_log, 'a', encoding='utf-8', newline='')
            wr = csv.writer(f)

        if 'multi_news' in data_label or 'cnn_dailymail'in data_label or 'gigaword' in data_label or 'samsum' in data_label or 'xsum' in data_label:
            self.log('sum rouge_score', rouge_score, prog_bar=True, logger=True)
            if self.args.output_log != None:    
                wr.writerow([self.args.dataset, self.args.prompt_name, self.args.scale, self.args.n_prefix,self.args.learning_rate, 'sum rouge_score', rouge_score])
        elif 'common_gen' in data_label or 'wiki_bio' in data_label:
            self.log('struct2txt rouge_score', rouge_score, prog_bar=True, logger=True)
            if self.args.output_log != None:    
                wr.writerow([self.args.dataset, self.args.prompt_name, self.args.scale, self.args.n_prefix,self.args.learning_rate, 'struct2txt rouge_score', rouge_score])
        elif 'imdb' in data_label or 'amazon_polarity' in data_label or 'rotten_tomatoes' in data_label or 'yelp_review_full' in data_label:
            self.log('sentiment accuracy', accuracy, prog_bar=True, logger=True)
            if self.args.output_log != None:    
                wr.writerow([self.args.dataset, self.args.prompt_name, self.args.scale, self.args.n_prefix,self.args.learning_rate, 'sentiment accuracy', accuracy])
        elif 'glue/qqp' in data_label or 'paws' in data_label or 'glue/mrpc' in data_label or self.args.dataset == 'glue':
            self.log('paraphrase accuracy', accuracy, prog_bar=True, logger=True)
            if self.args.output_log != None:    
                wr.writerow([self.args.dataset, self.args.prompt_name, self.args.scale, self.args.n_prefix,self.args.learning_rate, 'paraphrase accuracy', accuracy])
        elif "ag_news" in data_label or "dbpedia_14" in data_label or "trec" in data_label:
            self.log('topic accuracy', accuracy, prog_bar=True, logger=True)
            if self.args.output_log != None:    
                wr.writerow([self.args.dataset, self.args.prompt_name, self.args.scale, self.args.n_prefix,self.args.learning_rate, 'topic accuracy', accuracy])
        elif "quail" in data_label or "quartz" in data_label or "wiqa" in data_label or "wiki_hop" in data_label or "cos_e" in data_label or "dream" in data_label or "qasc" in data_label or "quarel" in data_label or "sciq" in data_label or "social_i_qa" in data_label or  "cosmos_qa" in data_label:
            self.log('multiqa accuracy', accuracy, prog_bar=True, logger=True)
            if self.args.output_log != None:    
                wr.writerow([self.args.dataset, self.args.prompt_name, self.args.scale, self.args.n_prefix,self.args.learning_rate, 'multiqa accuracy', accuracy])
        elif 'adversarial_qa/dbidaf' in data_label or 'adversarial_qa/dbert' in data_label or 'adversarial_qa/droberta' in data_label or 'quoref' in data_label:
            self.log('extractqa em', em_score, prog_bar=True, logger=True)
            if self.args.output_log != None:    
                wr.writerow([self.args.dataset, self.args.prompt_name, self.args.scale, self.args.n_prefix,self.args.learning_rate, 'extractqa em', em_score])
        elif self.args.dataset == 'super_glue' or self.args.dataset == 'anli': 
            self.log('accuracy', accuracy, prog_bar=True, logger=True)
            if self.args.output_log != None:    
                wr.writerow([self.args.dataset, self.args.prompt_name, self.args.scale, self.args.n_prefix,self.args.learning_rate, 'accuracy', accuracy])
        elif self.args.dataset == 'lama' or self.args.dataset == 'TriviaQA':
            self.log('em_score', em_score, prog_bar=True, logger=True)
            self.log('f1_score', f1_score, prog_bar=True, logger=True)
            if self.args.output_log != None:    
                wr.writerow([self.args.dataset, self.args.prompt_name, self.args.scale, self.args.n_prefix,self.args.learning_rate, 'em_score', em_score,'f1_score', f1_score])
        else:
            self.log('em_score', em_score, prog_bar=True, logger=True)
            if self.args.output_log != None:    
                wr.writerow([self.args.dataset, self.args.prompt_name, self.args.scale, self.args.n_prefix,self.args.learning_rate, 'em_score', em_score])
        if self.args.output_log != None:    
            f.close()        

    def configure_optimizers(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        if self.args.method == 'prompt_tune':
            print([n for n, p in model.named_parameters() if n == "shared.new_embed.weight"])
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if n == "shared.new_embed.weight"],
                    "weight_decay": 0.0,
                }
            ]
        else:
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
        if self.args.method != 'residual' and self.args.method!= 'prompt_tune':
            optimizer = Adafactor(optimizer_grouped_parameters, lr=self.args.learning_rate, scale_parameter=False, relative_step=False)
        else: 
            optimizer = AdamW(self.model.parameters(), lr=self.args.learning_rate)

        if self.args.use_lr_scheduling:
            len_data = len(self.train_dataloader())
            denomniator = (self.args.n_gpu * self.args.gradient_accumulation_steps)
            steps_per_epoch = ( len_data // denomniator ) + 1
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.args.learning_rate, steps_per_epoch=steps_per_epoch, pct_start=0.1, epochs=self.args.num_train_epochs, anneal_strategy='linear', cycle_momentum=False)
            return [optimizer], [{"scheduler": lr_scheduler, "interval": "step", "name": "learning rate"}]
        else:
            return [optimizer]

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("loss", loss)
        return loss

    def training_epoch_end(self, training_step_outputs):
        if self.args.method == 'residual' or self.args.method == 'prompt_tune':  
            MYDIR = ("/".join((self.args.output_dir.split('/'))[:-1]))
            CHECK_FOLDER = os.path.isdir(MYDIR)
            if not CHECK_FOLDER:
                os.makedirs(MYDIR)
                print("created folder : ", MYDIR)
            else:
                print(MYDIR, "folder already exists.")
            param_dict = {}
            # save only the trained adapter
            if self.args.method == 'residual':  
                for name, param in self.model.named_parameters():
                    if 'residual' in name:
                        param_dict[name]=param.clone().detach().cpu()
            elif self.args.method == 'prompt_tune':
                for name, param in self.model.named_parameters():
                    if 'new_embed' in name:
                        param_dict[name]=param.clone().detach().cpu()
            torch.save(param_dict, self.args.output_dir) 

    # def validation_epoch_end(self, validation_step_outputs):
    #     if 'evaluate' not in self.args.mode:
    #         MYDIR = ("/".join((self.args.output_dir.split('/'))[:-1]))
    #         CHECK_FOLDER = os.path.isdir(MYDIR)
    #         if not CHECK_FOLDER:
    #             os.makedirs(MYDIR)
    #             print("created folder : ", MYDIR)
    #         else:
    #             print(MYDIR, "folder already exists.")
    #         param_dict = {}
    #         # save only the trained adapter
    #         if self.args.method == 'residual' or self.args.method == 'token_adapter':  
    #             for name, param in self.model.named_parameters():
    #                 if 'residual' in name:
    #                     param_dict[name]=param.clone().detach().cpu()
    #         elif self.args.method == 'prompt_tune':
    #             for name, param in self.model.named_parameters():
    #                 if 'new_embed' in name:
    #                     param_dict[name]=param.clone().detach().cpu()
    #         torch.save(param_dict, self.args.output_dir)  
    
    def validation_step(self, batch, batch_idx):
        return self._generative_step(batch, batch_idx)

    def train_dataloader(self):
        # zerotune - t-zero replication setting
        if self.args.mode == 'zerotune':
            total_dataset = []
            for dataset in self.args.dataset:
                dataset_elem = self.get_dataset(dataset=dataset, tokenizer=self.tokenizer, type_path="train", args=self.args)
                total_dataset.append(dataset_elem)
            train_dataset = ConcatDataset(total_dataset)
        else:
            train_dataset = self.get_dataset(dataset=self.args.dataset, tokenizer=self.tokenizer, type_path="train", args=self.args)
        sampler = RandomSampler(train_dataset)
        dataloader = DataLoader(train_dataset, sampler=sampler,  batch_size=self.args.train_batch_size, drop_last=True, num_workers=self.args.num_workers)
        return dataloader

    def val_dataloader(self):
        # zerotune - t-zero replication setting
        if self.args.mode == 'zerotune':
            total_dataset = []
            for dataset in self.args.dataset:
                if dataset=='app_reviews':
                    continue
                dataset_elem = self.get_dataset(dataset=dataset, tokenizer=self.tokenizer, type_path="validation", args=self.args)
                total_dataset.append(dataset_elem)
            validation_dataset = ConcatDataset(total_dataset)
        else:
            validation_dataset = self.get_dataset(dataset=self.args.dataset, tokenizer=self.tokenizer, type_path="validation", args=self.args)
        return DataLoader(validation_dataset, batch_size=self.args.eval_batch_size, num_workers=self.args.num_workers, shuffle=False)
    
    def test_dataloader(self):
        test_dataset = self.get_dataset(dataset=self.args.dataset, tokenizer=self.tokenizer, type_path="test", args=self.args)
        
        return DataLoader(test_dataset, batch_size=self.args.eval_batch_size, num_workers=self.args.num_workers, shuffle=False)
