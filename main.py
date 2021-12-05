import os
import torch
import argparse
import pandas as pd
from torch import optim
import torch.nn as nn
from seq2seq import EncoderRNN, EncoderPQRNN, LuongAttnDecoderRNN, BeamSearchDecoder, trainIters, valIters, evaluateInput
from prado.data import create_dataloaders, Voc

# parser function
def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_path", default='data/index_without_duplicates.csv')
    parser.add_argument("--vocab_path", default='data/vocab.txt')
    parser.add_argument("--mode", default='train', choices=['train', 'test', 'val'])
    parser.add_argument("--use_cuda", action='store_true')
    parser.add_argument("--beam_width", default=5, type=int, help="beam width when doing beam search, the function will degrade to greedy search if the value is 1")
    parser.add_argument("--proj", action="store_true", help="Whether to use projection to generate word embedding")
    parser.add_argument('--max_length', default=32, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--emb_size', default=128, type=int)
    parser.add_argument('--hidden_size', default=256, type=int)
    parser.add_argument('--encoder_n_layers', default=2, type=int)
    parser.add_argument('--decoder_n_layers', default=2, type=int)
    parser.add_argument('--attn_model', default="general")
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--teacher_forcing_ratio', default=0.1, type=float, help="a value between 0.0 to 1.0, the larger the ratio, the higher the probability to use teacher forcing")
    parser.add_argument('--clip', default=50.0, type=float)
    parser.add_argument('--decoder_learning_ratio', default=5.0, type=float)
    parser.add_argument('--print_every', default=100, type=int, help="print the training loss every X iterations")
    parser.add_argument('--n_epoch', default=3, type=int)
    parser.add_argument('--checkpoint_iter', default=3000, type=int, help="Save model every X iterations")
    parser.add_argument('--load_model_path', default=None, help="Path of the model to be loaded")
    parser.add_argument("--result_path", default="test_results.csv")
    parser.add_argument("--save_dir", default="saved_models")
    parser.add_argument("--model_name", default='lstm')
    parser.add_argument("--corpus_name", default="customQA")
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parse()
    # Default word tokens
    PAD_token = 0  # Used for padding short sentences
    SOS_token = 1  # Start-of-sentence token
    EOS_token = 2  # End-of-sentence token
    voc = Voc(args.vocab_path, PAD_token, SOS_token, EOS_token)
    # # Set checkpoint to load from; set to None if starting from scratch
    loadFilename = args.load_model_path
    # checkpoint_iter = 4000
    USE_CUDA = torch.cuda.is_available() and args.use_cuda
    device = torch.device("cuda" if USE_CUDA else "cpu")
    # Load model if a loadFilename is provided
    if loadFilename:
        # If loading on same machine the model was trained on
        checkpoint = torch.load(loadFilename, map_location=device)
        # If loading a model trained on GPU to CPU
        # checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
        encoder_sd = checkpoint['en']
        decoder_sd = checkpoint['de']
        encoder_optimizer_sd = checkpoint['en_opt']
        decoder_optimizer_sd = checkpoint['de_opt']
        embedding_sd = checkpoint['embedding']
        voc.__dict__ = checkpoint['voc_dict']
    ## build saving directory
    directory = os.path.join(args.save_dir, args.corpus_name, args.model_name,
                             '{}-{}_{}'.format(args.encoder_n_layers, args.decoder_n_layers, args.hidden_size))
    if not os.path.exists(directory):
        os.makedirs(directory)

    ## Save the arguments to setting.txt
    argsDict = args.__dict__
    with open(os.path.join(directory, 'setting.txt'), 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')
    if args.mode == 'train':
        # Prepare dataset
        df = pd.read_csv(args.corpus_path)[:200000]
        # loadFilename = os.path.join(save_dir, model_name, corpus_name,
        #                            '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
        #                            '{}_checkpoint.tar'.format(checkpoint_iter))
        train_dataloader, test_dataloader = create_dataloaders(df, PAD_token, SOS_token, EOS_token, args.max_length, args.batch_size, proj_feature_size=args.hidden_size*2 if args.proj else None, voc=voc)

        print('Building encoder and decoder ...')
        if args.proj:
            embedding = nn.Embedding(voc.num_words, args.hidden_size)
            encoder = EncoderPQRNN(args.emb_size,args.hidden_size, args.encoder_n_layers, args.dropout)
            decoder = LuongAttnDecoderRNN(args.attn_model, embedding, args.hidden_size, voc.num_words, args.decoder_n_layers, args.dropout)
            if loadFilename:
                encoder.load_state_dict(encoder_sd)
                decoder.load_state_dict(decoder_sd)
            # Use appropriate device
            encoder = encoder.to(device)
            decoder = decoder.to(device)
            print('Models built and ready to go!')

            # Ensure dropout layers are in train mode
            encoder.train()
            decoder.train()
            # Initialize optimizers
            print('Building optimizers ...')
            encoder_optimizer = optim.Adam(encoder.parameters(), lr=args.learning_rate)
            decoder_optimizer = optim.Adam(decoder.parameters(), lr=args.learning_rate * args.decoder_learning_ratio)
            if loadFilename:
                encoder_optimizer.load_state_dict(encoder_optimizer_sd)
                decoder_optimizer.load_state_dict(decoder_optimizer_sd)

            # Run training iterations
            print("Starting Training!")
            trainIters(voc, train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer,
                       embedding, args.batch_size, args.n_epoch, SOS_token, directory,
                       args.print_every, args.clip, args.teacher_forcing_ratio, device=device, pqrnn=args.proj)
        else:
            # Initialize word embeddings
            embedding = nn.Embedding(voc.num_words, args.hidden_size)
            if loadFilename:
                embedding.load_state_dict(embedding_sd)
            # Initialize encoder & decoder models
            encoder = EncoderRNN(args.hidden_size, embedding, args.encoder_n_layers, args.dropout)
            decoder = LuongAttnDecoderRNN(args.attn_model, embedding, args.hidden_size, voc.num_words, args.decoder_n_layers, args.dropout)
            if loadFilename:
                encoder.load_state_dict(encoder_sd)
                decoder.load_state_dict(decoder_sd)
            # Use appropriate device
            encoder = encoder.to(device)
            decoder = decoder.to(device)
            print('Models built and ready to go!')

            # Ensure dropout layers are in train mode
            encoder.train()
            decoder.train()
            # Initialize optimizers
            print('Building optimizers ...')
            encoder_optimizer = optim.Adam(encoder.parameters(), lr=args.learning_rate)
            decoder_optimizer = optim.Adam(decoder.parameters(), lr=args.learning_rate * args.decoder_learning_ratio)
            if loadFilename:
                encoder_optimizer.load_state_dict(encoder_optimizer_sd)
                decoder_optimizer.load_state_dict(decoder_optimizer_sd)

            # Run training iterations
            print("Starting Training!")
            trainIters(voc, train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer,
                       embedding, args.batch_size, args.n_epoch, SOS_token, directory,
                       args.print_every, args.clip, args.teacher_forcing_ratio, device=device)
    if args.mode == 'val' or args.mode == 'train':
        if args.mode == 'val':
            # Prepare dataset
            df = pd.read_csv(args.corpus_path)
            train_dataloader, test_dataloader = create_dataloaders(df, PAD_token, SOS_token, EOS_token, args.max_length,
                                                                   args.batch_size,
                                                                   proj_feature_size=args.hidden_size * 2 if args.proj else None,
                                                                   voc=voc)
            # Initialize word embeddings
            embedding = nn.Embedding(voc.num_words, args.hidden_size)

            if loadFilename:
                embedding.load_state_dict(embedding_sd)
            # Initialize encoder & decoder models
            if args.proj:
                encoder = encoder = EncoderPQRNN(args.emb_size, args.hidden_size, args.encoder_n_layers, args.dropout)
            else:
                encoder = EncoderRNN(args.hidden_size, embedding, args.encoder_n_layers, args.dropout)
            decoder = LuongAttnDecoderRNN(args.attn_model, embedding, args.hidden_size, voc.num_words,
                                          args.decoder_n_layers, args.dropout)
            if loadFilename:
                encoder.load_state_dict(encoder_sd)
                decoder.load_state_dict(decoder_sd)
            # Use appropriate device
            encoder = encoder.to(device)
            decoder = decoder.to(device)
        # Validation
        # Set dropout layers to eval mode
        encoder.eval()
        decoder.eval()
        searcher = BeamSearchDecoder(encoder, decoder, device, SOS_token, EOS_token, args.batch_size, args.beam_width)
        original_sentences, decoded_sentences = valIters(voc, test_dataloader, searcher, device, pqrnn=args.proj)
        pd.DataFrame({'original_sentence': original_sentences, 'predicted_sentence':decoded_sentences}).to_csv(os.path.join(directory, args.result_path), index=False)

    if args.mode == 'test':
        # Test with a human input
        embedding = nn.Embedding(voc.num_words, args.hidden_size)
        print("Loading")
        if loadFilename:
            embedding.load_state_dict(embedding_sd)
        # Initialize encoder & decoder models
        if args.proj:
            encoder = encoder = EncoderPQRNN(args.emb_size,args.hidden_size, args.encoder_n_layers, args.dropout)
        else:
            encoder = EncoderRNN(args.hidden_size, embedding, args.encoder_n_layers, args.dropout)
        decoder = LuongAttnDecoderRNN(args.attn_model, embedding, args.hidden_size, voc.num_words,
                                      args.decoder_n_layers, args.dropout)
        if loadFilename:
            encoder.load_state_dict(encoder_sd)
            decoder.load_state_dict(decoder_sd)
        # Use appropriate device
        encoder = encoder.to(device)
        decoder = decoder.to(device)
        # Set dropout layers to eval mode
        encoder.eval()
        decoder.eval()
        searcher = BeamSearchDecoder(encoder, decoder, device, SOS_token, EOS_token, 1, args.beam_width)
        print("Loading complete, you can talk to me now!")
        evaluateInput(searcher, voc, max_length=args.max_length, device=device, proj_feature_size=args.hidden_size if args.proj else None)

