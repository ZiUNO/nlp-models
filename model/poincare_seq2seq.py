# -*- coding: utf-8 -*-
# @Time    : 2020/11/27 18:59
# @Author  : ZiUNO
# @Email   : ziunocao@126.com
# @File    : poincare_seq2seq.py
# @Software: PyCharm

from model import *


class PoincareSeq2Seq(Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, beam_size):
        super(PoincareSeq2Seq, self).__init__()
        """
        Embedding layer
            in: 1
            out: embedding_dim
            *default: padding_idx: 0
        """
        self.vocab_size = vocab_size
        self.beam_size = beam_size
        self.emb = Embedding(vocab_size, embedding_dim, padding_idx=0)
        """
        Seq2Seq layer
            in: embedding_dim
            out: embedding_dim
        """
        self.enc_cell = GRU(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True)
        self.dec_cell = GRU(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = Linear(hidden_dim, embedding_dim)
        """
        Pump layer
            in: embedding_dim
            out: vocab_size
            *default: pump.weight will be initialized to be equal to inverse of emb.weight
        """
        self.pum = Linear(embedding_dim, vocab_size, bias=False)
        # set pump's weight equal to the inverse of embed's weight
        self.pum.weight = Parameter(self.emb.weight.pinverse().transpose(0, 1))
        """
        Others
        """
        self.enc_seq_length = self.dec_seq_length = 0

    def forward(self, enc_input, dec_input):
        """
        Calculate target output
        :param enc_input: enc_input.shape = [batch_size, seq_length, embedding_dim]
        :param dec_input: dec_input.shape = [batch_size, seq_length + 1, embedding_dim]
                          # dec_input[batch_index, 0] = <START>.embedding
        :return: tgt_output.shape = [batch_size, seq_length + 1, embedding_dim]
                 # tgt_output[batch_index, -1] = <STOP>.embedding
        """
        self.enc_seq_length = max(enc_input.shape[1], self.enc_seq_length)
        self.dec_seq_length = max(dec_input.shape[1], self.dec_seq_length)
        # embedding enc_input & dec_input
        enc_emb = self.emb(enc_input)
        dec_emb = self.emb(dec_input)
        # get target output from Seq2Seq layer
        _, hidden_state = self.enc_cell(enc_emb)
        dec_output, _ = self.dec_cell(dec_emb, hidden_state)
        tgt_output = self.fc(dec_output)
        # pump embedding dim to vocab dim
        pump_output = self.pum(tgt_output)
        return pump_output

    def encode(self, enc_input):
        enc_emb = self.emb(enc_input)
        _, sentence_vec = self.enc_cell(enc_emb)
        return sentence_vec

    def decode(self, dec_input, sentence_vec):
        batch_size = dec_input.shape[0]
        out = self.emb(dec_input)[:, :1]  # [batch, 1, embedding_dim]
        dec_out, sentence_vec = self.dec_cell(out, sentence_vec)
        out = self.fc(dec_out)
        pum_out = self.pum(out[:, 0])

        beam_out, index = torch.topk(pum_out, self.beam_size, dim=-1)
        index = index.view(batch_size, self.beam_size, 1)
        output = (torch.log_softmax(beam_out, dim=-1), index,
                  sentence_vec.expand(self.beam_size, sentence_vec.shape[0], sentence_vec.shape[1],
                                      sentence_vec.shape[2]))
        with torch.no_grad():
            for length in range(self.dec_seq_length - 1):
                out, all_index, sentence_vec = output
                index = all_index[:, :, -1]
                tmp_output = torch.tensor([]).cuda()
                tmp_vec = torch.tensor([]).cuda()
                for i in range(self.beam_size):
                    beam_before_out = out[:, i: i + 1]
                    beam_index = index[:, i:i + 1]
                    beam_out = self.emb(beam_index)
                    dec_out, beam_sentence_vec = self.dec_cell(beam_out, sentence_vec[i])
                    dec_out = dec_out[:, 0]
                    fc_out = self.fc(dec_out)
                    pum_out = self.pum(fc_out)
                    pum_log_sum_out = torch.log_softmax(pum_out, dim=-1)
                    tmp_output = torch.cat([tmp_output, (pum_log_sum_out + beam_before_out.expand_as(pum_log_sum_out))],
                                           dim=1)
                    tmp_vec = torch.cat([tmp_vec, beam_sentence_vec], dim=0)
                tmp_output, tmp_index = tmp_output.topk(self.beam_size)
                tmp_vec = tmp_vec.unsqueeze(1).transpose(0, 2).transpose(1, 2)
                batch_output = torch.tensor([]).cuda()
                batch_source_index = torch.tensor([]).cuda()
                batch_target_index = torch.tensor([]).cuda()
                batch_vec = torch.tensor([]).cuda()
                for b in range(batch_size):
                    sample_output = torch.tensor([]).cuda()
                    sample_source_index = torch.tensor([]).cuda()
                    sample_target_index = torch.tensor([]).cuda()
                    sample_vec = torch.tensor([]).cuda()
                    for i in range(self.beam_size):
                        for j in range(self.beam_size):
                            if i * self.vocab_size <= tmp_index[b][j].tolist() < (i + 1) * self.vocab_size:
                                sample_output = torch.cat([sample_output, tmp_output[b][j].view(1)])
                                sample_source_index = torch.cat([sample_source_index, index[b][i].view(1)])
                                sample_target_index = torch.cat(
                                    [sample_target_index, tmp_index[b][j].view(1) % self.vocab_size])
                                sample_vec = torch.cat([sample_vec, tmp_vec[b][j:j + 1]])

                    batch_output = torch.cat([batch_output, sample_output], dim=0)
                    batch_source_index = torch.cat([batch_source_index, sample_source_index], dim=0)
                    batch_target_index = torch.cat([batch_target_index, sample_target_index], dim=0)
                    batch_vec = torch.cat([batch_vec, sample_vec], dim=0)
                batch_output = batch_output.reshape(batch_size, self.beam_size)
                batch_source_index = batch_source_index.reshape(batch_size, self.beam_size)
                batch_target_index = batch_target_index.reshape(batch_size, self.beam_size)
                batch_vec = batch_vec.reshape(batch_size, self.beam_size, batch_vec.shape[-2], batch_vec.shape[-1])
                tmp_output = torch.tensor([]).cuda()
                tmp_index = torch.tensor([]).cuda()
                tmp_vec = torch.tensor([]).cuda()
                for b in range(batch_size):
                    sample_output = batch_output[b]
                    sample_source_index = batch_source_index[b]
                    sample_target_index = batch_target_index[b]
                    sample_vec = batch_vec[b]
                    for i in range(self.beam_size):
                        for j in range(self.beam_size):
                            if int(sample_source_index[j].tolist()) == int(all_index[b][i][-1].tolist()):
                                sample_source_index[j] = -1
                                tmp_output = torch.cat([tmp_output, sample_output[j].view(1)])
                                tmp_index = torch.cat(
                                    [tmp_index, torch.cat([all_index[b][i].cuda(), sample_target_index[j].view(1)])])
                                tmp_vec = torch.cat([tmp_vec, sample_vec[j:j + 1]])
                tmp_output = tmp_output.reshape(batch_size, self.beam_size)
                tmp_vec = tmp_vec.reshape(batch_size,
                                          self.beam_size,
                                          tmp_vec.shape[-2],
                                          tmp_vec.shape[-1]).transpose(0, 2).transpose(0, 1).contiguous().cuda()
                tmp_index = tmp_index.reshape(batch_size, self.beam_size, length + 2).type(torch.long)
                output = (tmp_output, tmp_index, tmp_vec)
        final = torch.tensor([]).cuda()
        for b in range(batch_size):
            max_index = output[0][b].argmax()
            path = output[1][b][max_index]
            final = torch.cat([final, path])
        final = final.reshape(batch_size, self.dec_seq_length)
        output = final.clone().detach().type(torch.long)
        stop_index = self.vocab_size - 1
        for i in range(len(output)):
            for j in range(len(output[i])):
                if j == len(output[i]) - 2:
                    break
                if output[i][j].tolist() != stop_index:
                    continue
                output[i][j + 1:] = torch.tensor([0] * (len(output[i]) - j - 1))
                break
        return output
