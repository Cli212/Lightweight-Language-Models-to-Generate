# Lightweight-Language-Models-to-Generate

First, put the preprocessed in directory [data](./data), and use [this notebook](./data/remove_empty_and_duplicate_questions.ipynb) to remove the empty and duplicate lines from the processed data.

To train the model:

```shell
python main.py --corpus_path data/index_without_duplicates.csv --vocab_path data/vocab.txt --dropout 0.5 --learning_rate 5e-5 --batch_size 64 --emb_size 128 --hidden_size 256 --use_cuda --n_epoch 30 --model_name gru_seq2seq --save_dir saved_models
```

the trained model and checkpoints will be automatically saved in the \${save_dir}/\${corpus_name}/\${model_name}

or you can try to use [PQRNN](https://arxiv.org/pdf/2101.08890.pdf) as a encoder by activating projection (--proj):

```shell
python main.py --corpus_path data/index_without_duplicates.csv --vocab_path data/vocab.txt --proj
```

Training script will automaticlly split the corpus into train set and test set (8:2) and the test results will be saved in \${save_dir}/\${corpus_name}/\${model_name}.

Or you can also manually test the model by loading a saved checkpoint:

```shell
python main.py --corpus_path data/index_without_duplicates.csv --vocab_path data/vocab.txt --load_model_path saved-checkpoint-file --mode val
```

Besides, you can interactively have a conversation with the bot by:

```shell
python main.py --corpus_path data/index_without_duplicates.csv --vocab_path data/vocab.txt --load_model_path saved-checkpoint-file
--mode test
```

