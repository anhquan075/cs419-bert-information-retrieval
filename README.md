# BERT meets Cranfield
Recently, pre-trained BERT-based rankers achieved outstanding results on well-known collections. We investigate BERT-based rankers performance on the Cranfield collection, which comes with full relevance judgment on all documents in the collection. For more details, check out our paper:

- [BERT meets Cranfield: Uncovering the Properties of Full Ranking on Fully Labeled Data](https://djoerdhiemstra.com/wp-content/uploads/eacl2021swr.pdf)


# Requirements
Make sure to install all the neccessary packages using the following command:
```bash
pip3 install -r requirements.txt
```

# Data/
Cranfield collection was downloaded from [here](http://ir.dcs.gla.ac.uk/resources/test_collections/cran/).

You can also find the folds we used in this folder under /folds.

# Getting Started
You can set Hyper-parameters in the `ranker.py` file. 
- Mode: Can be set to 'Full-ranker' and 'Re-ranker'
- MODEL_TYPE: We used the bert-base-uncased model but you can easily change it to any model listed [here](https://huggingface.co/transformers/pretrained_models.html)
- LEARNING_RATE: We tested values 2e-5, 3e-5, 5e-5 for our experiments as suggested by [BERT](https://www.aclweb.org/anthology/N19-1423/) paper.
- MAX_LENGTH: Indicates the BERT token limitation. We just tested values 128, 256 due to hardware limitation.
- BATCH_SIZE: Indicates the batch size for fine-tuning. We tested the sizes 16, 32 as suggested by [BERT](https://www.aclweb.org/anthology/N19-1423/) paper.
- EPOCHS: Indicates the epoch numbers for fine-tuning. We tested 2, 3, 4 number of epochs  as suggested by [BERT](https://www.aclweb.org/anthology/N19-1423/) paper.

`ranker.py` will run the experiment with chosen parameters.

# Cite
If you use this work, please cite as:
```bash
@inproceedings{ghasemi-hiemstra-2021-bert,
    title = "{BERT} meets Cranfield: Uncovering the Properties of Full Ranking on Fully Labeled Data",
    author = "Ghasemi, Negin  and Hiemstra, Djoerd",
    booktitle = "Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Student Research Workshop",
    year = "2021",
    publisher = "Association for Computational Linguistics",
    pages = "58--64"
}
```

# Contact
If you have any questions, please contact Negin Ghasemi at N.Ghasemi@cs.ru.nl