import utils
import data_utils
from operator import itemgetter
from datetime import datetime
import os
import logging
import sys
import argparse
from tqdm import tqdm

import pickle
import torch

# For debugging
# os.environ["CUDA_VISIBLE_DEVICES"]="7"

os.makedirs("log_folder", exist_ok=True)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(formatter)

datetime_res = datetime.now().strftime('%H_%M_%d_%m_%Y')
file_handler = logging.FileHandler('./log_folder/{}.log'.format(datetime.now().strftime('mylogfile_{}'.format(datetime_res))))
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)


logger.addHandler(file_handler)
logger.addHandler(stdout_handler)

# ========================================
#               Hyper-Parameters
# ========================================
def init_args():
    parser = argparse.ArgumentParser(description='BERT CS419')
    parser.add_argument('--seed', type=int, default=76)
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--mode_pickle', type=str, default='rb')
    parser.add_argument('--model_type', type=str, default='amberoad/bert-multilingual-passage-reranking-msmarco')
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=3)
    
    parser.add_argument('--top_bm25', type=int, default=100)
    parser.add_argument('--map_cut', type=int, default=100)
    parser.add_argument('--ndcg_cut', type=int, default=100)
    parser.add_argument('--test_batch_size', type=int, default=1400)
    
    parser.add_argument('--fold_folder', type=str, default='../data/folds/')
    parser.add_argument('--output_model_path', type=str, default='output_model/{}'.format(datetime_res))
    parser.add_argument('--pretrained_model_path', type=str, default='./output_model/16_18_25_06_2022/model_epoch_2_step_100_loss_0.014_acc_97.31.pth')
    
    parser.add_argument('--corpus_file', type=str, default='../data/cran/cran_corpus.json')
    parser.add_argument('--query_file', type=str, default='../data/cran/cran_qry.json')
    parser.add_argument('--evaluation_file', type=str, default='../data/cran/cranqrel')
    
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = init_args()
    # Set the args.seed value all over the place to make this reproducible.
    utils.initialize_random_generators(args.seed)
    logger.info("============================================")
    logger.info("              Hyper-Parameters")
    logger.info("MODE: {}".format(args.mode))
    logger.info("MODEL_TYPE: {}".format(args.model_type))
    logger.info("LEARNING RATE: {}".format(args.learning_rate))
    logger.info("MAX LENGTH: {}".format(args.max_length))
    logger.info("BATCH SIZE: {}".format(args.batch_size))
    logger.info("EPOCHS: {}".format(args.epochs))
    logger.info("============================================")

    device = utils.get_gpu_device(logger)
    if not os.path.exists('../outputs_folder'):
        os.makedirs('../outputs_folder')

    queries = data_utils.get_queries(args.query_file)
    corpus = data_utils.get_corpus(args.corpus_file)
    rel_fed = data_utils.get_judgments(args.evaluation_file)

    labels = utils.get_binary_labels(rel_fed, len(queries), len(corpus))
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    tokenized_queries = [query.split(" ") for query in queries]

    bm25, bm25_top_n = utils.get_bm25_top_results(tokenized_corpus, tokenized_queries, args.top_bm25)

    if args.mode_pickle == 'wb':
        with open("../tokenizer_value_{}.pkl".format(args.max_length), args.mode_pickle) as fo:
            padded_all, attention_mask_all, token_type_ids_all, temp_feedback = utils.bert_tokenizer(corpus,
                                                                                                    queries,
                                                                                                    args.max_length, args.model_type)
            pickle.dump((padded_all, attention_mask_all, token_type_ids_all, temp_feedback), fo)
    elif args.mode_pickle == 'rb':
        with open("../tokenizer_value_{}.pkl".format(args.max_length), args.mode_pickle) as fo:
            padded_all, attention_mask_all, token_type_ids_all, temp_feedback = pickle.load(fo)
            logger.info('Load tokenizer successfully.')
    else:
        raise ValueError("Support rb and wb mode.")
    
    # ========================================
    #               Folds
    # ========================================
    mrr_bert_list, ap_11_interpolated_list, map_bert_list, ndcg_bert_list = [], [], [], []
    mrr_bert, ap_11_interpolated_bert, map_bert, ndcg_bert = 0, 0, 0, 0

    for fold_number in range(1, 6):
        logger.info('======== Fold {:} / {:} ========'.format(fold_number, 5))
        train_index, test_index = data_utils.load_fold(args.fold_folder, fold_number)

        padded, attention_mask, token_type_ids = [], [], []

        temp_feedback = []
        for query_num in tqdm(range(0, len(bm25_top_n))):
            if query_num in test_index:
                doc_nums = range(0, len(corpus))
            else:
                doc_nums = bm25_top_n[query_num]
            padded.append(list(itemgetter(*doc_nums)(padded_all[query_num])))
            attention_mask.append(list(itemgetter(*doc_nums)(attention_mask_all[query_num])))
            token_type_ids.append(list(itemgetter(*doc_nums)(token_type_ids_all[query_num])))
            temp_feedback.append(list(itemgetter(*doc_nums)(labels[query_num])))

        train_dataset = data_utils.get_tensor_dataset(train_index, padded, attention_mask, token_type_ids,
                                                    temp_feedback)
        test_dataset = data_utils.get_tensor_dataset(test_index, padded, attention_mask, token_type_ids, temp_feedback)

        train_dataloader, test_dataloader, model, optimizer, scheduler = utils.model_preparation(args.model_type, device, 
                                                                                                train_dataset, test_dataset,
                                                                                                args.batch_size, args.test_batch_size,
                                                                                                args.learning_rate, args.epochs)
        # ========================================
        #               Training Loop
        # ========================================
        if args.mode.lower() == 'train':
            epochs_train_loss, epochs_val_loss = [], []
            for epoch_i in range(0, args.epochs):
                # ========================================
                #               Training
                # ========================================
                logger.info('======== Epoch {:} / {:} ========'.format(epoch_i + 1, args.epochs))
                logger.info('Training...')
                model, optimizer, scheduler = utils.training(model, train_dataloader, device, optimizer, scheduler, logger, args.output_model_path, epoch_i)
        elif args.mode.lower() == 'test':
            if os.path.exists(args.pretrained_model_path):
                model.load_state_dict(torch.load(args.pretrained_model_path))
            else:
                raise ValueError('Found no {} path is exist. Please check again.'.format(args.pretrained_model_path))
        else:
            raise ValueError('No suitable mode, just support train or test mode currently.')
        # ========================================
        #               Testing
        # ========================================
        logger.info('Testing...')
        mrr_bert, ap_11_interpolated_bert, map_bert, ndcg_bert, mrr_bert_list, ap_11_interpolated_list, map_bert_list, ndcg_bert_list = utils.testing(model,
                                                                                                    test_dataloader,
                                                                                                    device, test_index,
                                                                                                    mrr_bert_list, 
                                                                                                    ap_11_interpolated_list,
                                                                                                    map_bert_list,
                                                                                                    ndcg_bert_list,
                                                                                                    mrr_bert, map_bert, ap_11_interpolated_bert,
                                                                                                    ndcg_bert, rel_fed,
                                                                                                    fold_number,
                                                                                                    args.map_cut, args.ndcg_cut, logger,
                                                                                                    len(corpus))

    logger.info("BERT MRR:  " + "{:.4f}".format(mrr_bert / 5))
    logger.info("BERT MAP:  " + "{:.4f}".format(map_bert / 5))
    logger.info("BERT NDCG: " + "{:.4f}".format(ndcg_bert / 5))

    os.makedirs("./results_256_token/{}".format(datetime_res), exist_ok=True)
    utils.results_to_csv('./results_256_token/{}/mrr_bert_{}_{}_{}_list.csv'.format(datetime_res, args.learning_rate, args.batch_size, args.epochs), mrr_bert_list)
    utils.results_to_csv('./results_256_token/{}/map_bert_{}_{}_{}_list.csv'.format(datetime_res, args.learning_rate, args.batch_size, args.epochs), map_bert_list)
    utils.results_to_csv('./results_256_token/{}/ndcg_bert_{}_{}_{}_list.csv'.format(datetime_res, args.learning_rate, args.batch_size, args.epochs), ndcg_bert_list)
