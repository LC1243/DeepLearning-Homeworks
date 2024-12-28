import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


def reshape_state(state):
    h_state = state[0]
    c_state = state[1]
    new_h_state = torch.cat([h_state[:-1], h_state[1:]], dim=2)
    new_c_state = torch.cat([c_state[:-1], c_state[1:]], dim=2)
    return (new_h_state, new_c_state)


class BahdanauAttention(nn.Module):
    """
    Bahdanau attention mechanism:
    score(h_i, s_j) = v^T * tanh(W_h h_i + W_s s_j)
    """

    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        
        self.Ws = nn.Linear(hidden_size, hidden_size)
        self.Wh = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)
        #raise NotImplementedError("Add your implementation.")

    def forward(self, query, encoder_outputs, src_lengths):
        """
        query:          (batch_size, max_tgt_len, hidden_size)
        encoder_outputs:(batch_size, max_src_len, hidden_size)
        src_lengths:    (batch_size)
        Returns:
            attn_out:   (batch_size, max_tgt_len, hidden_size) - attended vector
        """

        batch_size, max_src_len, hidden_size = encoder_outputs.size()
        max_tgt_len = query.size(1)

        scores = []
        for t in range(max_tgt_len):
            # Reshape the query tensor for broadcasting
            query_t = query[:, t, :].unsqueeze(1)  # (batch_size, 1, hidden_size)

            # Compute alignment scores
            score_et = self.v(torch.tanh(self.Ws(query_t) + self.Wh(encoder_outputs)))
            scores.append(score_et)

        # (batch_size, max_tgt_len, max_src_len)
        scores = torch.cat(scores, dim=1) 

        # Mask out padding tokens
        mask = self.sequence_mask(src_lengths).unsqueeze(1)  # (batch_size, 1, max_src_len)
        scores = scores.masked_fill(~mask, float('-inf'))  # Set scores of padded elements to -inf

        # Compute attention weights
        attn_weights = torch.softmax(scores, dim=-1)  # (batch_size, max_tgt_len, max_src_len)

        context_vector = torch.bmm(attn_weights, encoder_outputs) 

        # (batch_size, max_tgt_len, hidden_size)
        return context_vector
        #raise NotImplementedError("Add your implementation.")

    def sequence_mask(self, lengths):
        """
        Creates a boolean mask from sequence lengths.
        True for valid positions, False for padding.
        """
        batch_size = lengths.numel()
        max_len = lengths.max()
        return (torch.arange(max_len, device=lengths.device)
                .unsqueeze(0)
                .repeat(batch_size, 1)
                .lt(lengths.unsqueeze(1)))


class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        hidden_size,
        padding_idx,
        dropout,
    ):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size // 2
        self.dropout = dropout

        self.embedding = nn.Embedding(
            src_vocab_size,
            hidden_size,
            padding_idx=padding_idx,
        )
        self.lstm = nn.LSTM(
            hidden_size,
            self.hidden_size,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(self.dropout)

    def forward(
        self,
        src,
        lengths,
    ):
        # src: (batch_size, max_src_len)
        # lengths: (batch_size)
        #############################################
        # TODO: Implement the forward pass of the encoder
        # Hints:
        # - Use torch.nn.utils.rnn.pack_padded_sequence to pack the padded sequences
        #   (before passing them to the LSTM)
        # - Use torch.nn.utils.rnn.pad_packed_sequence to unpack the packed sequences
        #   (after passing them to the LSTM)
        #############################################
        
        # Step 1: Embed the input sequences
        embedded = self.embedding(src)  # (batch_size, max_src_len, hidden_size)

        # Step 2: Apply dropout
        embedded = self.dropout(embedded)

        # Step 3: Pack the sequences for the LSTM
        packed_embedded = pack(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)

        # Step 4: Pass through the LSTM
        packed_outputs, final_hidden = self.lstm(packed_embedded)

        # Step 5: Unpack the outputs
        encoder_outputs, _ = unpack(packed_outputs, batch_first=True)  # (batch_size, max_src_len, num_directions * hidden_size)

        return encoder_outputs, final_hidden
        

        #############################################
        # END OF YOUR CODE
        #############################################
        # enc_output: (batch_size, max_src_len, hidden_size)
        # final_hidden: tuple with 2 tensors
        # each tensor is (num_layers * num_directions, batch_size, hidden_size)
        
        # raise NotImplementedError("Add your implementation.")


class Decoder(nn.Module):
    def __init__(
        self,
        hidden_size,
        tgt_vocab_size,
        attn,
        padding_idx,
        dropout,
    ):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.tgt_vocab_size = tgt_vocab_size
        self.dropout = dropout

        self.embedding = nn.Embedding(
            self.tgt_vocab_size, self.hidden_size, padding_idx=padding_idx
        )

        self.dropout = nn.Dropout(self.dropout)
        self.lstm = nn.LSTM(
            self.hidden_size,
            self.hidden_size,
            batch_first=True,
        )

        self.attn = attn

        # Linear layer to combine context and LSTM output
        self.fc_out = nn.Linear(hidden_size * 2, hidden_size)

    def forward(
        self,
        tgt,
        dec_state,
        encoder_outputs,
        src_lengths,
    ):
        # tgt: (batch_size, max_tgt_len)
        # dec_state: tuple with 2 tensors
        # each tensor is (num_layers * num_directions, batch_size, hidden_size)
        # encoder_outputs: (batch_size, max_src_len, hidden_size)
        # src_lengths: (batch_size)
        # bidirectional encoder outputs are concatenated, so we may need to
        # reshape the decoder states to be of size (num_layers, batch_size, 2*hidden_size)
        # if they are of size (num_layers*num_directions, batch_size, hidden_size)
        if dec_state[0].shape[0] == 2:
            dec_state = reshape_state(dec_state)

        #############################################
        # TODO: Implement the forward pass of the decoder
        # Hints:
        # - the input to the decoder is the previous target token,
        #   and the output is the next target token
        # - New token representations should be generated one at a time, given
        #   the previous token representation and the previous decoder state
        # - Add this somewhere in the decoder loop when you implement the attention mechanism in 3.2:
        # if self.attn is not None:
        #     output = self.attn(
        #         output,
        #         encoder_outputs,
        #         src_lengths,
        #     )
        #############################################
        batch_size, max_tgt_len = tgt.size(0), tgt.size(1)
        outputs = torch.zeros(batch_size, max_tgt_len, self.hidden_size, device=tgt.device)

        # Initial input to the decoder is the start-of-sequence token (assumed to be the first token)
        input_t = tgt[:, 0].unsqueeze(1)  # (batch_size, 1)

        # Get all attention outputs at once
        attn_out_all = self.attn(tgt[:, :max_tgt_len, :], encoder_outputs, src_lengths)  # (batch_size, max_tgt_len, hidden_size)

        for t in range(max_tgt_len):
            # Step 1: Embed the input token
            input_t_embedded = self.dropout(self.embedding(input_t))  # (batch_size, 1, hidden_size)

            # Step 2: Pass the embedded input and previous hidden states to the LSTM
            output, dec_state = self.lstm(input_t_embedded, dec_state)  # output: (batch_size, 1, hidden_size)

            # Step 3: Calculate attention if applicable
            if self.attn is not None:
                 # Use the entire attention output for the current timestep
                attn_out = attn_out_all[:, t, :]  # Get the attention output for the current timestep

                # Concatenate attention output with LSTM output
                output = torch.cat((attn_out, output.squeeze(1)), dim=1)  # (batch_size, 2 * hidden_size)
                output = torch.tanh(self.fc_out(output))


            # Step 4: Store the output
            outputs[:, t, :] = output.squeeze(1)  # (batch_size, hidden_size)

            # Step 5: Prepare the next input (next token)
            # If we are in the last time step, we don't need to prepare next input.
            if t < max_tgt_len - 1:
                input_t = tgt[:, t + 1].unsqueeze(1)  # (batch_size, 1)
        
        return outputs, dec_state

        #############################################
        # END OF YOUR CODE
        #############################################
        # outputs: (batch_size, max_tgt_len, hidden_size)
        # dec_state: tuple with 2 tensors
        # each tensor is (num_layers, batch_size, hidden_size)
        
        # raise NotImplementedError("Add your implementation.")


class Seq2Seq(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
    ):
        super(Seq2Seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.generator = nn.Linear(decoder.hidden_size, decoder.tgt_vocab_size)

        self.generator.weight = self.decoder.embedding.weight

    def forward(
        self,
        src,
        src_lengths,
        tgt,
        dec_hidden=None,
    ):

        encoder_outputs, final_enc_state = self.encoder(src, src_lengths)

        if dec_hidden is None:
            dec_hidden = final_enc_state

        output, dec_hidden = self.decoder(
            tgt, dec_hidden, encoder_outputs, src_lengths
        )

        return self.generator(output), dec_hidden
