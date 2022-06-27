from genericpath import exists
import numpy as np
import torch
from scipy import stats
import csv 
from tqdm import tqdm
from rank_bm25 import BM25Okapi 
from operator import itemgetter
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler
from torch.optim import AdamW
import random
import os

def initialize_random_generators(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_gpu_device(logger):
    if torch.cuda.is_available():
        logger.info('GPU Type: {}'.format(torch.cuda.get_device_name(0)))
        return torch.device("cuda")
    else:
        logger.info('No GPU available, using the CPU instead.')
        return torch.device("cpu")


def reciprocal_rank(retrieved):
    retrieved = np.asarray(retrieved).nonzero()[0]
    return 1. / (retrieved[0] + 1) if retrieved.size else 0.


def precision_at_k(retrieved, k):
    assert k >= 1
    retrieved = np.asarray(retrieved)[:k] != 0
    if retrieved.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(retrieved)


def average_precision(retrieved, golds):
    retrieved = np.asarray(retrieved) != 0
    out = [precision_at_k(retrieved, k + 1) for k in range(retrieved.size) if retrieved[k]]
    if not out:
        return 0.
    return np.sum(out[:golds]) / golds

# [1, 1,0,1,0,0,0,0...]

def interpolated_11_point(retrieved, golds):
    retrieved = np.asarray(retrieved) != 0
    recall_score = [k/golds for k in range(1, len(np.argwhere(retrieved != 0)) + 1)]
    precision_score = [precision_at_k(retrieved, k + 1) for k in range(retrieved.size) if retrieved[k]]
    
    interplolated_values = np.linspace(0, 1, 11)
    interplolated_values = list(interplolated_values[::-1])
    rhoInterp = []
    recallValid = []
    # For each recallValues (0, 0.1, 0.2, ... , 1)
    for r in interplolated_values:
        # Obtain all recall values higher or equal than r
        argGreaterRecalls = np.argwhere(recall_score[:] >= r)
        pmax = 0
        # If there are recalls above r
        if argGreaterRecalls.size != 0:
            pmax = max(precision_score[argGreaterRecalls.min():])
        # print(r, pmax, argGreaterRecalls)
        recallValid.append(r)
        rhoInterp.append(pmax)
        
    ap = sum(rhoInterp) / 11        
    return ap


def dcg(retrieved, k):
    retrieved = np.asfarray(retrieved)[:k]
    if retrieved.size:
        return np.sum(retrieved / np.log2(np.arange(2, retrieved.size + 2)))


def ndcg(retrieved, gold_list, k):
    dcg_max = dcg(sorted(gold_list, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg(retrieved, k) / dcg_max


def retrieve_gold(rel_fed, query_num):
    res_list = [i for i in range(len(rel_fed)) if rel_fed[i][0] == query_num + 1]
    gold = [rel_fed[i][1:] for i in res_list]
    gold_unsorted = list(zip(*gold))[0]
    gold_sorted = sorted(gold, key=lambda l1: l1[1])
    gold_sorted_score = [5 - gs for gs in list(zip(*gold_sorted))[1]]
    return gold, gold_unsorted, gold_sorted_score


def get_binary_labels(rel_fed, length_of_queries_lst, length_of_corpus_lst):
    labels = []
    for query_num in range(0, length_of_queries_lst):
        gold, gold_unsorted, gold_sorted_score = retrieve_gold(rel_fed, query_num)
        current_labels = np.zeros(length_of_corpus_lst + 1)
        current_labels[list(gold_unsorted)] = 1
        current_labels = current_labels[1:]
        labels.append(current_labels)
    return labels


def get_bm25_top_results(tokenized_corpus, tokenized_queries, n):
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_top_n = []
    query_num = 0

    for query in tqdm(tokenized_queries, desc='Get BM25 results'):
        doc_scores = bm25.get_scores(query)
        feedback = list(zip(doc_scores, range(0, len(tokenized_corpus))))
        feedback_sorted = sorted(feedback, reverse=True)
        bm25_top_n.append(list(zip(*feedback_sorted))[1][:n])
        query_num += 1
    return bm25, bm25_top_n


def bert_tokenizer(corpus, queries, max_length, model_type):
    padded, attention_mask, token_type_ids = [], [], []
    tokenizer = AutoTokenizer.from_pretrained(model_type, do_lower_case=True,) #cache_dir=f'/vinai/quannla/bert-meets-cranfield/Code/pretrained-model/cache/{model_type}', local_files_only=True)
    #tokenizer.save_pretrained(f'/vinai/quannla/bert-meets-cranfield/Code/pretrained-model/{model_type}')
    temp_feedback = []

    for query_num in tqdm(range(0, len(queries))):
        temp_corpus = corpus

        current_padded, current_attention_mask, current_token_type_ids = [], [], []
        for document in temp_corpus:
            current_encoded = tokenizer(queries[query_num],
                                        document,
                                        truncation='only_second',
                                        max_length=max_length,
                                        pad_to_max_length=True,
                                        return_tensors='pt')
            current_padded.append(current_encoded['input_ids'])
            current_attention_mask.append(current_encoded['attention_mask'])
            current_token_type_ids.append(current_encoded['token_type_ids'])

        padded.append(current_padded)
        attention_mask.append(current_attention_mask)
        token_type_ids.append(current_token_type_ids)

    return padded, attention_mask, token_type_ids, temp_feedback


def mrr_ap_11_map_ndcg(trec_fold, current_query, query_num, feedback, rel_fed, mrr_total, ap_11_total, map_total, ndcg_total, mrr_list, ap_11_list,
                 map_list, ndcg_list, mode, fold_number, map_cut, ndcg_cut):
    feedback_sorted = sorted(feedback, reverse=True)

    walker = 1
    for fs in tqdm(feedback_sorted):
        current_fold = current_query
        current_fold += "D" + str(fs[1] + 1) + "\t" + str(walker) + "\t" + str(fs[0]) + "\t" + "run" + "\n"
        trec_fold += current_fold
        walker += 1

    text_file = open("../outputs_folder/result-" + mode + "-" + "Fold" + str(fold_number), "w")
    text_file.write(trec_fold)
    text_file.close()

    gold, gold_unsorted, gold_sorted_score = retrieve_gold(rel_fed, query_num)
    selected_candidates_id = list(zip(*feedback_sorted))[1]
    selected_candidates = [1 if s + 1 in gold_unsorted else 0 for s in selected_candidates_id]
    selected_candidates_sorted = [(5 - gold[gold_unsorted.index(s + 1)][1]) if (s + 1) in gold_unsorted else 0 for s in
                                  selected_candidates_id[:20]]

    mrr_current = reciprocal_rank(selected_candidates[:map_cut])
    mrr_list.append(mrr_current)
    mrr_total += mrr_current

    ap_11_interpolated_current = interpolated_11_point(selected_candidates[:map_cut], len(gold_unsorted))
    ap_11_list.append(ap_11_interpolated_current)
    ap_11_total += ap_11_interpolated_current

    map_current = average_precision(selected_candidates[:map_cut], len(gold_unsorted))
    map_list.append(map_current)
    map_total += map_current

    ndcg_current = ndcg(selected_candidates_sorted, gold_sorted_score, ndcg_cut)
    ndcg_list.append(ndcg_current)
    ndcg_total += ndcg_current

    return mrr_total, ap_11_total, map_total, ndcg_total, mrr_list, ap_11_list, map_list, ndcg_list, trec_fold


def model_preparation(MODEL_TYPE, device, train_dataset, test_dataset, batch_size, batch_size_test, learning_rate, epochs):
    train_dataloader = DataLoader(train_dataset,
                                  sampler=RandomSampler(train_dataset),
                                  batch_size=batch_size)

    test_dataloader = DataLoader(test_dataset,
                                 sampler=None,
                                 batch_size=batch_size_test)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_TYPE,
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False,
        #cache_dir=f'/vinai/quannla/bert-meets-cranfield/Code/pretrained-model/{MODEL_TYPE}',
        #local_files_only=True
    )
    #model.save_pretrained(f'/vinai/quannla/bert-meets-cranfield/Code/pretrained-model/{MODEL_TYPE}')
    torch.cuda.empty_cache()
    
    # model = torch.nn.DataParallel(model, device_ids=[i for i in range(torch.cuda.device_count())])
    model.to(device)

    optimizer = AdamW(model.parameters(),
                      lr=learning_rate,
                      eps=1e-8)

    total_steps = len(train_dataloader) * epochs

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)

    return train_dataloader, test_dataloader, model, optimizer, scheduler


def training(model, train_dataloader, device, optimizer, scheduler, logger, save_model_path, epoch_numb):
    total_train_loss = []
    correct = 0
    total = 0
    
    os.makedirs(save_model_path, exist_ok=True)
    
    model.train()

    with tqdm(train_dataloader, unit="batch") as tepoch:
        for step, batch in enumerate(tepoch):
            tepoch.set_description('Batch {}  of  {}.'.format(step, len(train_dataloader)))

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_input_token = batch[2].to(device)
            b_labels = batch[3].to(device)

            model.zero_grad()
            outputs = model(b_input_ids,
                                token_type_ids=b_input_token,
                                attention_mask=b_input_mask,
                                labels=b_labels)
            #print(outputs, b_labels)
            loss = outputs.loss
            total_train_loss.append(loss.item())
            
            logits = outputs.logits
            predictions = torch.argmax(logits.softmax(dim=-1), dim=1)
            total += b_labels.size(0)
            correct += (predictions == b_labels).sum().item()

            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            if step % 100 == 0:
                torch.save(model.state_dict(), f'{save_model_path}/model_epoch_{epoch_numb}_step_{step}_loss_{round(loss.item(), 3)}_acc_{round(100 * correct / total, 2)}.pth')
            
            tepoch.set_postfix(loss=loss.item(), accuracy= correct / total)
            logger.debug('Batch {}  of  {}. Loss: {:.4f}. Accuracy: {:.2f}%'.format(step, len(train_dataloader), loss.item(), 100 * correct / total))

    avg_train_loss = sum(total_train_loss) / len(train_dataloader)
    logger.info("Average training loss: {:.4f}".format(avg_train_loss))
    return model, optimizer, scheduler


def testing(model, test_dataloader, device, test_index, mrr_bert_list, ap_11_interpolated_list, map_bert_list, ndcg_bert_list,
            mrr_bert, ap_11_interpolated_bert, map_bert, ndcg_bert, rel_fed, fold_number, map_cut, ndcg_cut, logger, lenth_of_corpus):
    model.eval()
    predictions, true_labels = [], []
    walker = 0
    for batch in tqdm(test_dataloader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_input_token, b_labels = batch

        outputs_list = []
        with torch.no_grad():
            if len(b_input_ids[0]) > 100:
                for i in range(0, 14):
                    torch.cuda.empty_cache()
                    current_b_input_ids = b_input_ids[100 * i:100 * (i + 1)]
                    current_b_input_mask = b_input_mask[100 * i:100 * (i + 1)]
                    current_b_input_token = b_input_token[100 * i:100 * (i + 1)]
                    outputs_list.append(model(current_b_input_ids, token_type_ids=current_b_input_token,
                                              attention_mask=current_b_input_mask))
            else:
                outputs = model(b_input_ids, token_type_ids=b_input_token, attention_mask=b_input_mask)
        logits = []
        if len(outputs_list):
            # logits = torch.cat(outputs_list, dim=0)
            for i in range(len(outputs_list)):
                if len(logits) > 0:
                    logits = np.append(logits, outputs_list[i][0].detach().cpu().numpy(), axis=0)
                else:
                    logits = outputs_list[i][0].detach().cpu().numpy()
            #    logits = np. outputs_list[i][0].detach().cpu().numpy()
        else:
            logits = outputs[0]
            logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        predictions.append(logits)
        true_labels.append(label_ids)
        walker += 1

    walker = 0
    mrr_total, ap_11_total, map_total, ndcg_total = 0, 0, 0, 0
    mrr_list, ap_11_list, map_list, ndcg_list = [], [], [], []
    trec_fold = ""
    for query_index in test_index:
        current_query = "Q" + str(query_index + 1) + "\t" + "Q0" + "\t"
        doc_scores = list(zip(*predictions[walker]))[1]
        feedback = list(zip(doc_scores, range(0, lenth_of_corpus)))
        mrr_total, ap_11_total, map_total, ndcg_total, mrr_list, ap_11_list, map_list, ndcg_list, trec_fold = mrr_ap_11_map_ndcg(trec_fold,
                                                                                                  current_query,
                                                                                                  query_index, feedback,
                                                                                                  rel_fed, mrr_total, ap_11_total,
                                                                                                  map_total, ndcg_total,
                                                                                                  mrr_list, ap_11_list, map_list,
                                                                                                  ndcg_list, 'BERT',
                                                                                                  fold_number, map_cut,
                                                                                                  ndcg_cut)
        walker += 1

    mrr_bert_list += list(zip(mrr_list, test_index))
    ap_11_interpolated_list += list(zip(ap_11_list, test_index))
    map_bert_list += list(zip(map_list, test_index))
    ndcg_bert_list += list(zip(ndcg_list, test_index))

    mrr_bert += mrr_total / len(test_index)
    ap_11_interpolated_bert += ap_11_total / len(test_index)
    map_bert += map_total / len(test_index)
    ndcg_bert += ndcg_total / len(test_index)
    logger.info("Test MRR:  " + "{:.4f}".format(mrr_total / len(test_index)))
    logger.info("Test 11-points interpolated precision:  " + "{:.4f}".format(ap_11_total / len(test_index)))
    logger.info("Test MAP:  " + "{:.4f}".format(map_total / len(test_index)))
    logger.info("Test NDCG: " + "{:.4f}".format(ndcg_total / len(test_index)))
    logger.info(len(map_bert_list))
    return mrr_bert, map_bert, ndcg_bert, mrr_bert_list, map_bert_list, ndcg_bert_list


def t_test(bm25_list, bert_list, measure, logger):
    prediction_bm25 = list(zip(*bm25_list))[0]
    prediction_bert = list(zip(*bert_list))[0]
    _, p_value = stats.ttest_ind(prediction_bm25, prediction_bert)
    # logger.info("t-value " + measure + ": " + "{:.4f}".format(t_value))
    logger.info("p-value " + measure + ": " + "{:.4f}".format(p_value))


def results_to_csv(location, result_list):
    with open(location, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["ap_score", "index_id"])
        writer.writerows(result_list)
