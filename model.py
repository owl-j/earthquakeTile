# -*- coding: utf-8 -*-
"""
Created on Wed May  5 16:35:03 2021

@author: Hong
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
meant to imitate the model after the predicting earthquake occurence model

but meant to modify it into something else with symbolic components later

start from the input layer

attention is implemented from https://blog.floydhub.com/attention-mechanism/
encoder decoder model adapted from https://github.com/bentrevett/pytorch-seq2seq/blob/master/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb


'''
class Attention(nn.Module):
    def __init__(self, hidden_size, method="dot"):
        super(Attention, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        
        if method == "concat":
            self.fc = nn.Linear(hidden_size, hidden_size, bias=False)
            self.w = nn.Parameter(torch.FloatTensor(1, hidden_size))
    def forward(self, decoder_hid, encoder_outputs):
        #actually dont know how the alignment scores are calculated.
        #but since they said they used FFNN structure, we are using the concat one
        if self.method == "dot":
            return encoder_outputs.bmm(decoder_hid.view(1,-1,1)).squeeze(-1)
        
        elif self.method == "concat":
            out = F.tanh(self.fc(decoder_hid + encoder_outputs))
            return out.bmm(self.w.unsqueeze(-1)).squeeze(-1)
    
class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim):
        super().__init__()
        
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        
        # directly put in the inputs.
        #dont know if the rnn cell used for encoding is LSTM or not but most of the examples i see uses LSTM.
        self.rnn = nn.LSTM(input_size=self.input_dim,
                           hidden_size=self.hid_dim,
                           num_layers=1,
                           bias=True,
                           dropout=0,
                           bidirectional=False)
        
    def forward(self, input_batch, hiddens=None):
        
        #input_batch = [src len, batch size]
        #hiddens consists of (last_hidden_weight, last_cell_state)
        
        if hiddens != None:
            output, (new_hiddens, cell) = self.rnn(input_batch, hiddens)
        else:
            output, (new_hiddens, cell) = self.rnn(input_batch)
        
        #outputs = [src len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #outputs are always from the top hidden layer
        return output, (new_hiddens, cell)

class Decoder(nn.Module):
    def __init__(self, input_hid_dim, hid_dim, attention="concat"):
        super(Decoder, self).__init__()
        
        #using multiplicative (Luong) attention to calc the score,
        #in our case we use dot product attention score
        
        #decoder is for one unit of input only.
        
        self.input_hid_dim = input_hid_dim
        self.hid_dim = hid_dim
        self.attention = Attention(hidden_size=hid_dim,method=attention)
        
        self.rnn = nn.LSTM(input_size=self.input_hid_dim,
                           hidden_size=self.hid_dim,
                           num_layers=1,
                           bias=True,
                           dropout=0,
                           bidirectional=False)
        
    def forward(self, input_batch, hiddens, encoder_outputs):
        #TODO: CHECK if the procedure matches (most of) the original model
        rnn_output = None 
        new_hidden = None
        cell = None
        
        rnn_output, (new_hidden, cell) = self.rnn(input_batch, hiddens)
        #calc scores
        alignment_scores = self.attention(rnn_output, encoder_outputs)
        #Softmax weights
        att_weights = F.softmax(alignment_scores.view(1,-1), dim=1)
        #make context vector, note that the context vector mentioned in the paper is the encoder_outputs variable in our case
        #and below is a amalgamated context vector
        context_vec = torch.bmm(att_weights.unsqueeze(0), encoder_outputs)
        # we will do a weighted sum with the input sequence with weight each comp. of
        #context vector
        # Concatenating output from LSTM with context vector
        output = torch.cat((rnn_output, context_vec),-1)
        ##TODO: maybe add a softmax layer here to the output. BUT need to understand what they want first
        return output, new_hidden, att_weights


class bd_attention_LSTM(nn.Module):
    def __init__(self, num_input, first_LSTM_size, 
                 biLSTM_first_size, biLSTM_second_size,
                 att_width, att_hid, h1_size, h2_size,
                 h3_size, h4_size, num_output):
        '''
        architecture:
            1.input layer
            2.one layer one directino LSTM
            3.two bidirectional LSTM layers stacked on top of each other
            4.self attention layer
            5.flatten layer
            6.four layers of feed forward neurons, with tanh as activation layers
        '''
        super(bd_attention_LSTM, self).__init__()
        self.num_input = num_input
        self.first_LSTM_size = first_LSTM_size
        self.biLSTM_first_size = biLSTM_first_size
        self.biLSTM_second_size = biLSTM_second_size
        self.att_width = att_width # is how many prev inputs we will use to attend the current input
        self.att_hid = att_hid #number of hidden neurons of RNN(LSTM)
        self.h1_size = h1_size
        self.h2_size = h2_size
        self.h3_size = h3_size
        self.h4_size = h4_size
        self.num_output = num_output
        
        self.LSTM_layer_1 = nn.LSTM(input_size=self.num_input,
                                    hidden_size=self.first_LSTM_size, num_layers=1,
                                    bias=True,dropout=0, bidirectional=False)
        
        self.bi_LSTM_layer_2 = nn.LSTM(input_size=self.first_LSTM_size,
                                       hidden_size=self.biLSTM_first_size,
                                       num_layers=1, bias=True, dropout=0,
                                       bidirectional=True)
        
        self.bi_LSTM_layer_3 = nn.LSTM(input_size=(2*self.biLSTM_first_size),
                                       hidden_size=self.biLSTM_second_size,
                                       num_layers=1, bias=True, dropout=0,
                                       bidirectional=True)
        #at this point the vector dimension is biLSTM_second_size, along with tuple consists of (hidden_weight (output), cell states (c))

        #my understanding is that the attention width is how many LSTM units we have for the whole encoding or decoding sequence
        #and the "100" (which the paper didnt mention what that is) is the number of hidden neurons we have.
        self.encoder = Encoder((2*self.biLSTM_second_size), self.att_hid, attention="concat")
        self.decoder = Decoder(input_hid_dim=self.att_hid, 
                               hid_dim=self.att_hid, attention="concat")
        
        #after this, the size of the input will be 
        self.flatten = nn.Sequential(nn.Flatten())
        
        self.W1 = nn.Linear(self.att_hid * self.att_width, h1_size)
        self.W2 = nn.Linear(h1_size, h2_size)
        self.W3 = nn.Linear(h2_size, h3_size)
        self.W4 = nn.Linear(h3_size, h4_size)
        self.out = nn.Linear(h4_size, num_output)
        
    def forward(self, input_seqs):
        target_seq_len = input_seqs.shape[0]
        assert(target_seq_len == self.att_width)
        #input_seqs dim : [total_inputs, batch_size, input_dim]
        #input structure: [[x_0, y_0], [x_1, y_1],..., [x_n, y_n]]. x,y: two sequences of inputs, n = total_inputs
        
        out, (hid, cell) = self.LSTM_layer_1(input_seqs)
        #now out dimension: [total_inputs, batch_size, hid_dim(ofLSTM_layer_1)]
        out, (hid2, cell2) = self.bi_LSTM_layer_2(out)
        #now out dimension: [total_inputs, batch_size, 2*hid_dim(ofLSTM_layer_2)]
        #Assumed that the two output from the two direction of training concated together
        out, (hid3, cell3) = self.bi_LSTM_layer_3(out)
        
        
        #now out dimension: [total_inputs, batch_size, 2*hid_dim(ofLSTM_layer_3)]
        enc_out, enc_hids = self.Encoder(out) #already did the recurring steps, but decoder needs to be fed in one by one since there are more steps
        
        #now enc_out is dimension: [total_inputs, batch_size, hid_dim(of Encoder)]
        #enc_out is the future context vector
        #for next pass and so on, we will feed the context_vector and the prev hidden state of the preceeding decoder
        
        #TODO: i dont know what the first input should be so I put in each encoder output as the input.
        decoder_input = enc_out
        decoder_hiddens = enc_hids
        decoder_outputs = []
        for i in range(target_seq_len):
            #decoder_input = enc_out[i].view(1, *enc_out[i].shape)
            decoder_input, decoder_hiddens, att_weights = self.decoder(decoder_input, decoder_hiddens, enc_out)
            #TODO: We can record the attention weights along the way but it was not implemented yet 
            #decoder_outputs.append(decoder_input.view(*decoder_output.shape[1:]).tolist())
            decoder_outputs.append(decoder_input)
            #decoder outputs dimension: [total_inputs, batch_size, att_hid]
        
        #TODO: not sure if this flattening method is correct since it was not mentioned in the paper
        decoder_outputs = torch.tensor(decoder_outputs)
        decoder_outputs = decoder_outputs.transpose(0,1)
        decoder_outputs = self.flatten(decoder_outputs) #now each element output of the same sequence of the same batch are concat together
        #now dim: [batch_size, att_hid*total_inputs]
        output = F.tanh(self.W1(decoder_outputs))
        output = F.tanh(self.W2(output))
        output = F.tanh(self.W3(output))
        output = F.tanh(self.W4(output))
        output = self.out(output)
        #TODO: depending on the task, add softmax layer, and add appropriate loss func in the training func
        return output
  
#TODO: mentioned non-determinism issue from Pytorch
#https://docs.nvidia.com/gameworks/content/developertools/desktop/environment_variables.htm
# https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html<<change the enviro var for the problem stated in there>>
