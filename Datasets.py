from torch.utils.data import Dataset
import pandas as pd
import json
import random
import ast
import torch

from datasets import load_dataset
from promptsource.templates import DatasetTemplates

class Pretrain(Dataset):
    def __init__(self, dataset, tokenizer, type_path, input_length, output_length, args):
        self.args = args
        self.tokenizer = tokenizer
        self.type_path = type_path
        self.dataset_name = dataset
        self.n_prefix = self.args.n_prefix
        self.dataset_config_name = self.args.dataset_config
        self.csv_read_flag=0
        ids_to_answers = None    
        if self.dataset_name== 'TriviaQA':
            kilt_triviaqa = load_dataset("kilt_tasks", name="triviaqa_support_only")
            trivia_qa = load_dataset('trivia_qa', 'unfiltered.nocontext')
            triviaqa_map = {}
            def add_missing_data(x, trivia_qa_subset, triviaqa_map):
                i = triviaqa_map[x['id']]
                x['input'] = trivia_qa_subset[i]['question']
                return x
            for k in ['train', 'validation', 'test']:
                triviaqa_map = dict([(q_id, i) for i, q_id in enumerate(trivia_qa[k]['question_id'])])
                kilt_triviaqa[k] = kilt_triviaqa[k].filter(lambda x: x['id'] in triviaqa_map)
                kilt_triviaqa[k] = kilt_triviaqa[k].map(add_missing_data, fn_kwargs=dict(trivia_qa_subset=trivia_qa[k], triviaqa_map=triviaqa_map))
            self.dataset = kilt_triviaqa[type_path]   
            with open('data/tqa_val_answers.json') as f:
                ids_to_answers = json.load(f)
        # dataset not used for training (anli, story cloze, superglue, hellaswag, winogrande, lambada)
        elif self.dataset_name == 'anli':
            self.dataset = load_dataset(self.dataset_name, split=self.dataset_config_name)
            self.prompt = DatasetTemplates('anli')
        elif self.dataset_name == 'story_cloze':
            self.dataset = load_dataset("story_cloze","2016", data_dir='data')['validation']
            self.prompt = DatasetTemplates('story_cloze', '2016')
        elif self.type_path == 'validation' and (self.dataset_name == 'super_glue' or self.dataset_name == 'hellaswag' or self.dataset_name == 'winogrande' or 'lambada' in self.dataset_name):
            if self.dataset_name == 'craffel/openai_lambada':
                type_path = 'test'
            self.dataset = load_dataset(self.dataset_name, self.dataset_config_name)[type_path]
            self.prompt = DatasetTemplates(
                f"{self.dataset_name}"
                if self.dataset_config_name is None
                else f"{self.dataset_name}/{self.dataset_config_name}"
            )
        # dataset used for training
        else:
            if self.type_path == 'train':
   
                unshuffled = load_dataset(self.dataset_name, self.dataset_config_name,ignore_verifications=True)[type_path]
                if (self.dataset_name == 'wiki_qa' and (self.args.prompt_name == "Jeopardy style" or self.args.prompt_name == "Topic Prediction - Question and Answer Pair" or self.args.prompt_name == "Generate Question from Topic" or self.args.prompt_name == "Topic Prediction - Question Only" or self.args.prompt_name == "Topic Prediction - Answer Only" or self.args.prompt_name == "Direct Answer to Question")) or (self.dataset_config_name == 'mrpc' and (self.args.prompt_name == "generate_paraphrase" or self.args.prompt_name == 'generate_sentence')) or self.args.prompt_name == 'paraphrase-task':
                    unshuffled = unshuffled.filter(lambda example: example["label"]==1)
                if (self.dataset_name == 'duorc' and (self.args.prompt_name == "generate_question_by_answer" or self.args.prompt_name == 'build_story_around_qa')):
                    unshuffled = unshuffled.filter(lambda example: example["no_answer"]==False)
                if len(unshuffled)>10000:
                    shuffled = unshuffled.shuffle(seed=40)
                    self.dataset=shuffled.select(range(10000))
                else:
                    self.dataset=unshuffled
                #self.dataset = unshuffled
            else: 
                # 5000 example for validation
                if self.dataset_name == 'wiki_bio' or self.dataset_name == 'imdb' or self.dataset_name == 'ag_news' or self.dataset_name == 'amazon_polarity' or self.dataset_name == 'yelp_review_full' or self.dataset_name == 'dbpedia_14' or self.dataset_name == 'trec' or self.dataset_name == 'craffel/openai_lambada':
                    type_path = 'test'
                unshuffled = load_dataset(self.dataset_name, self.dataset_config_name,ignore_verifications=True)[type_path]
                if self.dataset_name == 'wiki_hop':
                    for i in range(len(unshuffled)):
                        example_batch = unshuffled[i]
                        example_batch['annotations'] = []
                if (self.dataset_name == 'wiki_qa' and (self.args.prompt_name == "Jeopardy style" or self.args.prompt_name == "Topic Prediction - Question and Answer Pair" or self.args.prompt_name == "Generate Question from Topic" or self.args.prompt_name == "Topic Prediction - Question Only" or self.args.prompt_name == "Topic Prediction - Answer Only" or self.args.prompt_name == "Direct Answer to Question")) or (self.dataset_config_name == 'mrpc' and (self.args.prompt_name == "generate_paraphrase" or self.args.prompt_name == 'generate_sentence')) or self.args.prompt_name == 'paraphrase-task':
                    unshuffled = unshuffled.filter(lambda example: example["label"]==1)
                if len(unshuffled)>self.args.valid_data_size:
                    shuffled = unshuffled.shuffle(seed=40)
                    self.dataset=shuffled.select(range(self.args.valid_data_size))
                else:
                    self.dataset=unshuffled
            self.prompt = DatasetTemplates(
                f"{self.dataset_name}"
                if self.dataset_config_name is None
                else f"{self.dataset_name}/{self.dataset_config_name}"
            )
        print(f'Length of dataset retrieving is.. {len(self.dataset)}')
        self.input_length = input_length
        self.output_length = output_length
        self.ids_to_answers = ids_to_answers

    def __len__(self):
        return len(self.dataset)

    def convert_to_features(self, example_batch, index=None):
        # prompt evaluation
        options = 0
        labels = None
        if self.type_path == 'validation':
            if self.dataset_name== 'TriviaQA':
                input_ = example_batch['input']
                target_ = example_batch['output'][0]['answer']
                labels = example_batch['id']
            elif self.dataset_name== 'WOW':
                input_ = "Generate a informative response to the following dialogue.\n" + example_batch['input']
                target_ = example_batch['output'][0]['answer']
            elif self.dataset_name== 'lama':
                input_pre = example_batch['input']
                if self.args.prompt_name == 'please next word':
                    for index, word in enumerate(input_pre.split()):
                        if word == '<extra_id_0>':
                            input_pre = ' '.join(input_pre.split()[:index])
                            break
                    input_ = 'Please predict an entity after the following chunk of text. ' + input_pre
                elif self.args.prompt_name == 'complete sentence':
                    for index, word in enumerate(input_pre.split()):
                        if word == '<extra_id_0>':
                            input_pre = ' '.join(input_pre.split()[:index])
                            break
                    input_ = "Complete the sentence with an entity: " + input_pre
                target_ = example_batch['output']
            else:
                prompt_name = self.args.prompt_name
                prompt = self.prompt[prompt_name]
                result = prompt.apply(example_batch)
                input_ = result[0]
                target_ = result[1]
                if 'lambada' not in self.dataset_name:
                    options = prompt.get_answer_choices_list(example_batch)
        
        elif self.type_path == 'train':
            if self.dataset_name== 'TriviaQA':
                input_ = example_batch['input']
                target_ = example_batch['output'][0]['answer']
                labels = example_batch['id']
            elif self.dataset_name == 'lama':
                input_pre = example_batch['input']
                for index, word in enumerate(input_pre.split()):
                    if word == '<extra_id_0>':
                        input_pre = ' '.join(input_pre.split()[:index])
                        break
                input_ = input_pre
                target_ = example_batch['output']
            else:
                prompt_name = self.args.prompt_name
                prompt = self.prompt[prompt_name]
                result = prompt.apply(example_batch)
                input_ = result[0]
                target_ = result[1]
                options = prompt.get_answer_choices_list(example_batch)
     
        # To match T0 training & evaluation setting, we do not add EOS 
        if self.args.eos_token == False: 
            source = self.tokenizer.batch_encode_plus([str(input_)], max_length=self.input_length, 
                                                            padding='max_length', truncation=True, return_tensors="pt", add_special_tokens=False)
        else: 
             source = self.tokenizer.batch_encode_plus([str(input_)], max_length=self.input_length, 
                                                            padding='max_length', truncation=True, return_tensors="pt", add_special_tokens=True)
        targets = self.tokenizer.batch_encode_plus([str(target_)], max_length=self.output_length, 
                                                        padding='max_length', truncation=True, return_tensors="pt")
        
        # prepend_task_tokens
        if self.args.method == 'prompt_tune':
            task_tokens = ["<TASK{}>".format(str(i).zfill(2)) for i in range(self.n_prefix)]
            # self.tokenizer.add_tokens(task_tokens)
            task_token_ids = self.tokenizer(" ".join(task_tokens), return_tensors="pt", add_special_tokens=False)["input_ids"]
            assert task_token_ids.shape[-1]==self.n_prefix

            n_train = source["input_ids"].shape[0]
            new_input_ids=torch.cat([task_token_ids.repeat(n_train, 1),source["input_ids"]], 1)
            source["input_ids"] = new_input_ids
            # print(source["input_ids"])
            source["attention_mask"] = torch.cat([torch.ones((n_train, self.n_prefix), dtype=torch.long), source["attention_mask"]], 1)
 
        # data_label is needed for validation of T0 replication (multi-task learning)
        data_label = self.dataset_name   
        return source, targets, data_label, options, labels
  
    def __getitem__(self, index):
        if self.csv_read_flag==1:
            source, targets, data_label, options, labels = self.convert_to_features(self.dataset.iloc[index])
        else:
            source, targets, data_label, options, labels = self.convert_to_features(self.dataset[index])
        source_ids = source["input_ids"].squeeze()
        target_ids = targets["input_ids"].squeeze()

        src_mask    = source["attention_mask"].squeeze()
        target_mask = targets["attention_mask"].squeeze()

        if options is not None:
            option_list = options
        else:
            option_list = -1

        if labels is not None:
            label_ids = labels
        else:
            label_ids = -1

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask, "data_label": data_label, "option_list": option_list, "label_ids": label_ids}