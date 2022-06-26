import utils
import data_utils
from operator import itemgetter
from datetime import datetime
import os
import logging
import sys
from tqdm import tqdm

import pickle



os.makedirs("log_folder", exist_ok=True)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(formatter)

file_handler = logging.FileHandler('./log_folder/{}.log'.format(datetime.now().strftime('mylogfile_%H_%M_%d_%m_%Y')))
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)


logger.addHandler(file_handler)
logger.addHandler(stdout_handler)

# ========================================
#               Hyper-Parameters
# ========================================
SEED = 76
MODE = 'Full-ranker'
MODE_PICKLE = 'rb'
MODEL_TYPE = '/vinai/quannla/bert-meets-cranfield/Code/pretrained-model/amberoad/bert-multilingual-passage-reranking-msmarco'
LEARNING_RATE = 3e-5
MAX_LENGTH = 256
BATCH_SIZE = 64
EPOCHS = 3
TOP_BM25 = 100
MAP_CUT = 100
NDCG_CUT = 20
TEST_BATCH_SIZE = 1400

output_model_path = 'output_model/{}'.format(datetime.now().strftime('%H_%M_%d_%m_%Y'))

# Set the seed value all over the place to make this reproducible.
utils.initialize_random_generators(SEED)

if __name__ == "__main__":
    logger.info("============================================")
    logger.info("              Hyper-Parameters")
    logger.info("MODE: {}".format(MODE))
    logger.info("MODEL_TYPE: {}".format(MODEL_TYPE))
    logger.info("LEARNING RATE: {}".format(LEARNING_RATE))
    logger.info("MAX LENGTH: {}".format(MAX_LENGTH))
    logger.info("BATCH SIZE: {}".format(BATCH_SIZE))
    logger.info("EPOCHS: {}".format(EPOCHS))
    logger.info("============================================")

    device = utils.get_gpu_device(logger)
    if not os.path.exists('../outputs_folder'):
        os.makedirs('../outputs_folder')

    queries = data_utils.get_queries('/vinai/quannla/bert-meets-cranfield/Data/cran/cran_qry.json')
    corpus = data_utils.get_corpus('/vinai/quannla/bert-meets-cranfield/Data/cran/cran_corpus.json')
    rel_fed = data_utils.get_judgments('/vinai/quannla/bert-meets-cranfield/Data/cran/cranqrel')

    labels = utils.get_binary_labels(rel_fed)
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    tokenized_queries = [query.split(" ") for query in queries]

    bm25, bm25_top_n = utils.get_bm25_top_results(tokenized_corpus, tokenized_queries, TOP_BM25)

    if MODE_PICKLE == 'wb':
        with open("/vinai/quannla/bert-meets-cranfield/tokenizer_value_{}.pkl".format(MAX_LENGTH), MODE_PICKLE) as fo:
            padded_all, attention_mask_all, token_type_ids_all, temp_feedback = utils.bert_tokenizer(corpus,
                                                                                                    queries,
                                                                                                    MAX_LENGTH, MODEL_TYPE)
            pickle.dump((padded_all, attention_mask_all, token_type_ids_all, temp_feedback), fo)
    elif MODE_PICKLE == 'rb':
        with open("/vinai/quannla/bert-meets-cranfield/tokenizer_value_{}.pkl".format(MAX_LENGTH), MODE_PICKLE) as fo:
            padded_all, attention_mask_all, token_type_ids_all, temp_feedback = pickle.load(fo)
    else:
        raise ValueError("Support rb and wb mode.")
    
    # ========================================
    #               Folds
    # ========================================
    mrr_bm25_list, map_bm25_list, ndcg_bm25_list = [], [], []
    mrr_bert_list, map_bert_list, ndcg_bert_list = [], [], []
    mrr_bm25, map_bm25, ndcg_bm25 = 0, 0, 0
    mrr_bert, map_bert, ndcg_bert = 0, 0, 0

    for fold_number in range(1, 6):
        logger.info('======== Fold {:} / {:} ========'.format(fold_number, 5))
        train_index, test_index = data_utils.load_fold(fold_number)

        padded, attention_mask, token_type_ids = [], [], []

        temp_feedback = []
        for query_num in range(0, len(bm25_top_n)):
            if query_num in test_index:
                doc_nums = range(0, 1400)
            else:
                doc_nums = bm25_top_n[query_num]
            padded.append(list(itemgetter(*doc_nums)(padded_all[query_num])))
            attention_mask.append(list(itemgetter(*doc_nums)(attention_mask_all[query_num])))
            token_type_ids.append(list(itemgetter(*doc_nums)(token_type_ids_all[query_num])))
            temp_feedback.append(list(itemgetter(*doc_nums)(labels[query_num])))

        train_dataset = data_utils.get_tensor_dataset(train_index, padded, attention_mask, token_type_ids,
                                                    temp_feedback)
        test_dataset = data_utils.get_tensor_dataset(test_index, padded, attention_mask, token_type_ids, temp_feedback)

        train_dataloader, test_dataloader, model, optimizer, scheduler = utils.model_preparation(MODEL_TYPE, device, 
                                                                                                train_dataset, test_dataset,
                                                                                                BATCH_SIZE, TEST_BATCH_SIZE,
                                                                                                LEARNING_RATE, EPOCHS)
        # ========================================
        #               Training Loop
        # ========================================
        epochs_train_loss, epochs_val_loss = [], []
        for epoch_i in range(0, EPOCHS):
            # ========================================
            #               Training
            # ========================================
            logger.info('======== Epoch {:} / {:} ========'.format(epoch_i + 1, EPOCHS))
            logger.info('Training...')
            model, optimizer, scheduler = utils.training(model, train_dataloader, device, optimizer, scheduler, logger, output_model_path, epoch_i)
        # ========================================
        #               Testing
        # ========================================
        logger.info('Testing...')
        mrr_bert, map_bert, ndcg_bert, mrr_bert_list, map_bert_list, ndcg_bert_list = utils.testing(model,
                                                                                                    test_dataloader,
                                                                                                    device, test_index,
                                                                                                    mrr_bert_list,
                                                                                                    map_bert_list,
                                                                                                    ndcg_bert_list,
                                                                                                    mrr_bert, map_bert,
                                                                                                    ndcg_bert, rel_fed,
                                                                                                    fold_number,
                                                                                                    MAP_CUT, NDCG_CUT, logger)

    logger.info("BERT MRR:  " + "{:.4f}".format(mrr_bert / 5))
    logger.info("BERT MAP:  " + "{:.4f}".format(map_bert / 5))
    logger.info("BERT NDCG: " + "{:.4f}".format(ndcg_bert / 5))

    os.makedirs("results_256", exist_ok=True)
    utils.results_to_csv('./results_256/mrr_bert_{}_{}_{}_list.csv'.format(LEARNING_RATE, BATCH_SIZE, EPOCHS), mrr_bert_list)
    utils.results_to_csv('./results_256/map_bert_{}_{}_{}_list.csv'.format(LEARNING_RATE, BATCH_SIZE, EPOCHS), map_bert_list)
    utils.results_to_csv('./results_256/ndcg_bert_{}_{}_{}_list.csv'.format(LEARNING_RATE, BATCH_SIZE, EPOCHS), ndcg_bert_list)
