import os
import torch
import numpy as np

from tqdm import tqdm
from sklearn.metrics import f1_score, recall_score

from model_util import get_optimizer_and_scheduler, get_dataloader

def train(logger, model, inputs, batch_size, output_dir,
          dev_data, val_inputs,
          learning_rate=1e-5,
          warmup_steps=50,
          num_training_steps=200,
          gradient_accumulation_steps=1,
          max_grad_norm=1.0,
          eval_period=20,
          prompt_tune=False,
          head_tune=False,
          transform_tune=False,
          weight=False):
    optimizer, scheduler = get_optimizer_and_scheduler(
        "adafactor",
        model,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        num_training_steps=num_training_steps)
    dataloader = get_dataloader(inputs, batch_size, is_training=True)

    n_trainable_params = len([param for param in model.parameters() if param.requires_grad])
    n_gpus = torch.cuda.device_count()
    logger.info("Training {} parameters on {} examples for {} steps using {} GPUs".format(
        n_trainable_params, len(inputs["input_ids"]), num_training_steps, n_gpus))

    model.train()
    global_step = 0
    train_losses = []
    best_accuracy = -1
    stop_training=False

    logger.info("Start training")

    best_f1 = 0
    patience = 0
    for epoch in range(num_training_steps):
        for batch in dataloader:
            global_step += 1

            input_ids=batch[0].cuda()
            attention_mask=batch[1].cuda()
            token_type_ids=batch[2].cuda()
            if len(batch)==3:
                labels=None
            else:
                labels=batch[3].cuda()

            loss = run_model(model, input_ids, attention_mask, token_type_ids, labels=labels)
            loss = loss.mean()

            if torch.isnan(loss).data:
                print ("Stop training because loss=%s" % (loss.data))
                stop_training=True
                break
            train_losses.append(loss.detach().cpu())
            loss.backward()
            if global_step % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()    # We have accumulated enought gradients
                model.zero_grad()
                if scheduler is not None:
                    scheduler.step()

            if global_step % eval_period == 0:
                f1 = evaluate(model, dev_data, val_inputs, batch_size)
                logger.info("Performance at step=%d (train loss %.2f): %.2f" % \
                            (global_step, np.mean(train_losses), f1*100))

                if f1 >= best_f1:
                    best_f1 = f1
                    if prompt_tune:
                        keys = ["shared.new_embed.weight"]
                        model_state_dict = {key: model.state_dict()[key if n_gpus==1 else "module."+key].cpu() for key in keys}
                    elif head_tune:
                        keys = ["lm_head.my_lm_head.weight"]
                        model_state_dict = {key: model.state_dict()[key if n_gpus==1 else "module."+key].cpu() for key in keys}
                    elif transform_tune:
                        keys = ["lm_head.transform.weight"]
                        model_state_dict = {key: model.state_dict()[key if n_gpus==1 else "module."+key].cpu() for key in keys}
                    else:
                        model_state_dict = {k:v.cpu() for (k, v) in model.state_dict().items()}
                    torch.save(model_state_dict,
                            os.path.join(output_dir, "model-{}.pt".format(global_step)))
                    logger.info("Saving model at global_step=%d (train loss %.2f, f1 %.2f)" % \
                                (global_step, np.mean(train_losses), best_f1*100))
                    patience = 0
                else:
                    patience += 1
                
                train_losses = []
                if patience >= 6: 
                    break
                

            if global_step==num_training_steps:
                break

        if global_step==num_training_steps:
            break

    logger.info("Finish training")

def inference(model, inputs, batch_size, return_logits=False):
    dataloader = get_dataloader(inputs, batch_size, is_training=False)

    all_losses = []
    for batch in tqdm(dataloader):
        input_ids=batch[0].cuda()
        attention_mask=batch[1].cuda()
        token_type_ids=batch[2].cuda()

        if len(batch)==3:
            labels=None
        else:
            labels=batch[3].cuda()

        with torch.no_grad():

            loss = run_model(model, input_ids, attention_mask, token_type_ids,
                             labels=labels, return_logits=return_logits)

        all_losses += loss.cpu().detach().numpy().tolist()

    return all_losses

def evaluate(model, dev_data, val_inputs, batch_size):
    label_losses = []
    for input_tensor in val_inputs:
        label_losses.append(inference(model,
                                input_tensor,
                                batch_size))
    label_losses = {str(i): loss for i, loss in enumerate(label_losses)}
    labels = list(label_losses.keys())
    preds = []
    answers = []
    for idx, (_, label) in enumerate(dev_data):
        label_loss = {l:np.sum(label_losses[l][idx]) for l in label_losses}
        prediction = sorted(label_loss.items(), key=lambda x: x[1])[0][0]
        preds.append(prediction)
        answers.append(label)
    f1 = f1_score(answers, preds, pos_label='1')
    recall = recall_score(answers, preds, average=None)
    print("Recall:",  recall)
    return f1


def run_model(model, input_ids, attention_mask, token_type_ids,
              labels=None, return_logits=False, weight=False):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                    decoder_input_ids=input_ids,
                    decoder_attention_mask=None,
                    labels=None)
    logits = outputs.logits[..., :-1, :].contiguous()

    if return_logits:
        softmax = torch.nn.Softmax(dim=-1)
        return -torch.log(softmax(logits))

    if labels is None:
        labels = input_ids
    labels = labels[..., 1:].contiguous()
    label_mask = token_type_ids[..., 1:].contiguous()

    if weight:
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none", weight=[0.2, 0.8])
    else:
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

    losses = loss_fct(logits.view(-1, logits.size(-1)),
                      labels.view(-1)) # [batch_size, length]
    losses = losses.view(logits.size(0), logits.size(1)) * label_mask
    return torch.sum(losses, axis=1) / torch.sum(label_mask, axis=1)

