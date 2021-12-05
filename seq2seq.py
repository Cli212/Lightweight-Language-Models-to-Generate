import os
import re
import random
import operator
import unicodedata
import numpy as np
import torch
import pandas as pd
from queue import PriorityQueue
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
try:
    from tensorboardX import SummaryWriter
except ImportError:
    from torch.utils.tensorboard import SummaryWriter
from prado.model import PQRNN
from prado.data import create_dataloader_from_sentence


class EncoderRNN(nn.Module):
    """
    An encoder with RNN as the backbone model
    """
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True, batch_first=True)
        # use orthogonal init for GRU layer0 weights
        torch.nn.init.orthogonal(self.gru.weight_ih_l0)
        torch.nn.init.orthogonal(self.gru.weight_hh_l0)
        # use zero init for GRU layer0 bias
        # self.gru.bias_ih_l0.zero_()
        # self.gru.bias_hh_l0.zero_()

    def forward(self, input_seq, input_lengths, hidden=None):
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)
        # Pack padded batch of sequences for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, enforce_sorted=False, batch_first=True)
        # Forward pass through GRU
        outputs, hidden = self.gru(packed, hidden)
        # Unpack padding
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        # Return output and final hidden state
        return outputs, hidden


class EncoderPQRNN(nn.Module):
    """
    The projection-based models encoder.
    """
    def __init__(self, hidden_size, emb_size, n_layers=1, dropout=0):
        super(EncoderPQRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        # Initialize PQRNN; the b and d params are set to hyperparameters hidden_size and emb_size
        self.model = PQRNN(b=hidden_size, d=emb_size, num_layers=n_layers, dropout=dropout, output_size=hidden_size,
                           rnn_type="GRU", multilabel=False, nhead=2)

    def forward(self, projection, hidden=None):
        return self.model(projection, hidden)


# Luong attention layer
class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden.transpose(0, 1) * encoder_output.transpose(0, 1), dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden.transpose(0, 1) * energy.transpose(0, 1), dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(
            torch.cat((hidden.transpose(0, 1).expand(encoder_output.size(0), -1, -1), encoder_output.transpose(0, 1)),
                      2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)


class LuongAttnDecoderRNN(nn.Module):
    """
    This is the decoder class that includes attention model, embedding model and decoder
    """
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout),
                          batch_first=True)
        # use orthogonal init for GRU layer0 weights
        torch.nn.init.orthogonal(self.gru.weight_ih_l0)
        torch.nn.init.orthogonal(self.gru.weight_hh_l0)
        # use zero init for GRU layer0 bias
        # self.gru.bias_ih_l0.zero_()
        # self.gru.bias_hh_l0.zero_()
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        # Note: we run this one step (word) at a time
        # Get embedding of current input word
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        # Forward through unidirectional GRU
        rnn_output, hidden = self.gru(embedded, last_hidden)
        # Calculate attention weights from the current GRU output
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context = attn_weights.bmm(encoder_outputs)
        # Concatenate weighted context vector and GRU output using Luong eq. 5
        rnn_output = rnn_output.squeeze(1)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # Predict next word using Luong eq. 6
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        # Return output and final hidden state
        return output, hidden


def maskNLLLoss(inp, target, mask, device):
    """
     This loss function calculates the average negative log likelihood of the elements that correspond to a 1 in the mask tensor.
    :param inp:
    :param target:
    :param mask:
    :param device:
    :return:
    """
    # loss_fn = nn.CrossEntropyLoss()
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()


def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, SOS,
          encoder_optimizer, decoder_optimizer, batch_size, teacher_forcing_ratio, clip, device):
    """
    Training function for one iteration
    """
    # Zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Set device options
    input_variable = input_variable.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)
    # Lengths for rnn packing should always be on the cpu
    lengths = lengths.to("cpu")

    # Initialize variables
    loss = 0
    print_losses = []
    n_totals = 0

    # Forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[SOS] for _ in range(batch_size)])
    decoder_input = decoder_input.to(device)

    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    # Determine if we are using teacher forcing this iteration
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # Forward batch of sequences through decoder one time step at a time
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # Teacher forcing: next input is current target
            decoder_input = target_variable[t].view(1, -1).T
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t], device)
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # No teacher forcing: next input is decoder's own current output
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0]] for i in range(batch_size)])
            decoder_input = decoder_input.to(device)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t], device)
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    # Perform backpropatation
    loss.backward()

    # Clip gradients: gradients are modified in place
    _ = nn.utils.clip_grad_norm_(encoder.parameters(), clip, error_if_nonfinite=True)
    _ = nn.utils.clip_grad_norm_(decoder.parameters(), clip, error_if_nonfinite=True)

    # Adjust model weights
    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals


def train_pqrnn(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, SOS,
                encoder_optimizer, decoder_optimizer, batch_size, teacher_forcing_ratio, clip, device):
    """
    training function for prado and pqrnn that used projection EncoderRNN
    """
    # Zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Set device options
    input_variable = input_variable.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)
    # Lengths for rnn packing should always be on the cpu
    lengths = lengths.to("cpu")

    # Initialize variables
    loss = 0
    print_losses = []
    n_totals = 0

    # Forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable)

    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[SOS] for _ in range(batch_size)])
    decoder_input = decoder_input.to(device)

    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    # Determine if we are using teacher forcing this iteration
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # Forward batch of sequences through decoder one time step at a time
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # Teacher forcing: next input is current target
            decoder_input = target_variable[t].view(1, -1).T
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t], device)
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # No teacher forcing: next input is decoder's own current output
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0]] for i in range(batch_size)])
            decoder_input = decoder_input.to(device)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t], device)
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    # Perform backpropatation
    loss.backward()

    # Clip gradients: gradients are modified in place
    _ = nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Adjust model weights
    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals


def trainIters(voc, dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, embedding, batch_size, n_epoch, SOS, directory, print_every, clip,
               teacher_forcing_ratio, device, pqrnn=False):
    """
    The train functions that train n_epoch of all the data iteratively
    """
    # Load batches for each iteration
    # training_batches = [batch2TrainData(voc, [random.choice(pairs) for _ in range(batch_size)])
    # for _ in range(n_iteration)]

    # Initializations
    print('Initializing ...')
    iter_count = 0
    print_loss = 0
    # if loadFilename:
    #     start_iteration = checkpoint['iteration'] + 1
    # Save checkpoint

    writer = SummaryWriter(log_dir=os.path.join(directory, 'training_process'))
    # Training loop
    print("Training...")
    for epoch in range(n_epoch):
        for iteration, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}")):
            # training_batch = training_batches[iteration - 1]
            # Extract fields from batch
            iter_count += 1
            input_variable, lengths, target_variable, mask, max_target_len = batch

            # Run a training iteration with batch
            if pqrnn:
                loss = train_pqrnn(input_variable, lengths, target_variable.T, mask.T, max_target_len, encoder, decoder,
                                   SOS,
                                   encoder_optimizer, decoder_optimizer, batch_size, teacher_forcing_ratio, clip,
                                   device)
            else:
                loss = train(input_variable, lengths, target_variable.T, mask.T, max_target_len, encoder, decoder, SOS,
                             encoder_optimizer, decoder_optimizer, batch_size, teacher_forcing_ratio, clip, device)
            writer.add_scalar('scalar/loss', loss, iter_count)
            print_loss += loss
            # Print progress
            if iteration != 0 and iteration % print_every == 0:
                print_loss_avg = print_loss / print_every
                print("Epoch: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(epoch + 1, iteration / len(
                    dataloader) * 100, print_loss_avg))
                print_loss = 0
        torch.save({
            'epoch': epoch,
            'en': encoder.state_dict(),
            'de': decoder.state_dict(),
            'en_opt': encoder_optimizer.state_dict(),
            'de_opt': decoder_optimizer.state_dict(),
            'loss': loss,
            'voc_dict': voc.__dict__,
            'embedding': embedding.state_dict()
        }, os.path.join(directory, '{}_{}.tar'.format(epoch, 'checkpoint')))


def val(voc, searcher, input_variable, lengths, target_variable, mask, max_target_len, device, pqrnn):
    """
    The inference function, every time accept a batch of data and out put the generated tokens
    """
    # Set device options
    input_variable = input_variable.to(device)
    # target_variable = target_variable.to(device)
    mask = mask.to(device)
    # Lengths for rnn packing should always be on the cpu
    lengths = lengths.to("cpu")
    # Decode sentence with searcher
    tokens = searcher(input_variable, lengths, pqrnn)
    decoded_tokens = [i[0] for i in tokens]

    # indexes -> words
    original_tokens = target_variable.tolist()

    return original_tokens, decoded_tokens


def valIters(voc, dataloader, searcher, device, pqrnn=False):
    """
    Infer all the data iteratively
    """
    # Load batches for each iteration
    # training_batches = [batch2TrainData(voc, [random.choice(pairs) for _ in range(batch_size)])
    # for _ in range(n_iteration)]
    original_tokens = []
    decoded_tokens = []
    # Training loop
    print("Validating ...")
    for iteration, batch in enumerate(tqdm(dataloader, desc=f"Validation")):
        # training_batch = training_batches[iteration - 1]
        # Extract fields from batch
        input_variable, lengths, target_variable, mask, max_target_len = batch

        o_tokens, d_tokens = val(voc, searcher, input_variable, lengths, target_variable, mask, max_target_len, device,
                                 pqrnn)
        assert len(o_tokens) == len(d_tokens)
        original_tokens.extend(o_tokens)
        decoded_tokens.extend(d_tokens)
        # original_sentence, decoded_sentence = ' '.join(original_words), ' '.join(decoded_words)
        # df_list.append(pd.DataFrame({'original':[original_sentence], 'predicted':[decoded_sentence]}))
    # Return result
    return token2word(original_tokens, voc), token2word(decoded_tokens, voc)


def token2word(token_list, voc):
    """
    Transform token indexes to words/sub words
    :param token_list: token list with shape [batch_size, sequence_length]
    :param voc: vocabulary
    :return:
    """
    results = []
    for tokens in token_list:
        word_list = []
        for token in tokens:
            if voc.index2word[token].startswith('▁'):
                word_list.append(voc.index2word[token].lstrip('▁'))
            elif voc.index2word[token] == 'EOS':
                break
            else:
                if word_list == []:
                    word_list.append(voc.index2word[token])
                else:
                    word_list[-1] = word_list[-1] + voc.index2word[token]
        results.append(' '.join(word_list).lstrip("SOS "))
    return results


class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length):
        '''
        :param hiddenstate:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        '''
        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward
        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward

    def __lt__(self, other):
        return self.leng < other.leng

    def __gt__(self, other):
        return self.leng > other.leng


class BeamSearchDecoder(nn.Module):
    """
    A beam search decoder, when the beam width is 1, the decoder will degrade to a greedy search decoder.
    """
    def __init__(self, encoder, decoder, device, SOS_token, EOS_token, batch_size, beam_width=10):
        super(BeamSearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.SOS_token = SOS_token
        self.EOS_token = EOS_token
        self.batch_size = batch_size
        self.beam_width = beam_width

    def forward(self, input_seq, input_length, pqrnn=False):
        topk = 1
        decoded_batch = []
        # Forward input through encoder model
        if pqrnn:
            encoder_outputs, encoder_hidden = self.encoder(input_seq)
        else:
            encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hiddens = encoder_hidden[:-self.decoder.n_layers*2 if self.decoder.gru.bidirectional else self.decoder.n_layers]
        # Initialize decoder input with SOS_token
        # decoding goes sentence by sentence
        for idx in range(self.batch_size):
            decoder_hidden = decoder_hiddens[:, idx, :].unsqueeze(1)
            encoder_output = encoder_outputs[idx, :, :].unsqueeze(0)
            # Start with the start of the sentence token
            decoder_input = torch.LongTensor([[self.SOS_token]])
            decoder_input = decoder_input.to(self.device)
            # Number of sentence to generate
            endnodes = []
            number_required = min((topk + 1), topk - len(endnodes))

            # starting node -  hidden vector, previous node, word id, logp, length
            node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1)
            nodes = PriorityQueue()

            # start the queue
            nodes.put((-node.eval(), node))
            qsize = 1
            # start beam search
            while True:
                # give up when decoding takes too long
                if qsize > 2000: break

                # fetch the best node
                score, n = nodes.get()
                decoder_input = n.wordid
                decoder_hidden = n.h.contiguous()

                if n.wordid.item() == self.EOS_token and n.prevNode != None:
                    endnodes.append((score, n))
                    # if we reached maximum # of sentences required
                    if len(endnodes) >= number_required:
                        break
                    else:
                        continue

                # decode for one step using decoder
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_output)

                # PUT HERE REAL BEAM SEARCH OF TOP
                log_prob, indexes = torch.topk(decoder_output, self.beam_width)
                nextnodes = []

                for new_k in range(self.beam_width):
                    decoded_t = indexes[0][new_k].view(1, -1)
                    log_p = log_prob[0][new_k].item()

                    node = BeamSearchNode(decoder_hidden, n, decoded_t, n.logp + log_p, n.leng + 1)
                    score = -node.eval()
                    nextnodes.append((score, node))

                # put them into queue
                for i in range(len(nextnodes)):
                    score, nn = nextnodes[i]
                    nodes.put((score, nn))
                    # increase qsize
                qsize += len(nextnodes) - 1

            # choose nbest paths, back trace them
            if len(endnodes) == 0:
                endnodes = [nodes.get() for _ in range(topk)]

            utterances = []
            for score, n in sorted(endnodes, key=operator.itemgetter(0)):
                utterance = []
                utterance.append(n.wordid.item())
                # back trace
                while n.prevNode != None:
                    n = n.prevNode
                    utterance.append(n.wordid.item())

                utterance = utterance[::-1]
                utterances.append(utterance)

            decoded_batch.append(utterances)
        return decoded_batch

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )



def evaluateInput(searcher, voc, max_length, device, proj_feature_size=None):
    """
    The function that accepts a input from command line and generate a response
    :param searcher:
    :param voc:
    :param max_length:
    :param device:
    :param proj_feature_size:
    :return:
    """
    from bpemb import BPEmb
    bpemb_en = BPEmb(lang="en", dim=100)
    while (1):
        try:
            # Get input sentence
            input_sentence = input('> ')
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit': break
            # Normalize sentence
            input_sentence = normalizeString(input_sentence)
            # words -> indexes
            index_batch = [voc.word2index[word] for word in bpemb_en.encode(input_sentence)]
            dataloader = create_dataloader_from_sentence([(index_batch, index_batch)], voc, max_length, proj_feature_size*2 if proj_feature_size else None)
            # Evaluate sentence
            original_tokens = []
            decoded_tokens = []
            for iteration, batch in enumerate(dataloader):
                # Extract fields from batch
                input_variable, lengths, target_variable, mask, max_target_len = batch

                o_tokens, d_tokens = val(voc, searcher, input_variable, lengths, target_variable, mask, max_target_len,
                                         device,
                                         pqrnn=True if proj_feature_size else False)
                assert len(o_tokens) == len(d_tokens)
                original_tokens.extend(o_tokens)
                decoded_tokens.extend(d_tokens)

            print('Bot:', token2word(decoded_tokens, voc)[0])
        except KeyError:
            print("Error: Encountered unknown word.")
