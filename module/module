import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

class Myloss(nn.Module):
    def __init__(self, sqrting=False):
        super(Myloss, self).__init__()
        self.sqrting = sqrting

    def forward(self, input, output):
        "Remove the diagonal element to calculate loss "
        input = torch.mul(input, 1-(torch.eye(input.shape[-1])))
        output = torch.mul(output, 1-(torch.eye(input.shape[-1])))
        if self.sqrting:
            mask = torch.tensor(output > 0.0944, dtype=torch.float)
            output = torch.mul(output, mask)
            output = torch.sqrt(output)
        loss = F.mse_loss(input, output, reduction="mean")
        return loss

class MLP(nn.Module):
    """Two-layer fully-connected ELU net with batch norm."""
    def __init__(self, n_in, n_hid, n_out, do_prob=0.):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        self.bn = nn.BatchNorm1d(n_out)
        self.dropout_prob = do_prob  # dropout 概率

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        x = inputs.view(inputs.size(0)*inputs.size(1),-1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)

    def forward(self, inputs):
        # Input shape: [num_sims, num_things, num_fe  atures]
        x = F.elu(self.fc1(inputs))
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = F.elu(self.fc2(x))
        return self.batch_norm(x)

class rnn_base(nn.Module):
    def __init__(self,input_size,n_hid,layer,channels,time_step = None,issplit = False,gap = None,model_name = 'gru'):
        super(rnn_base, self).__init__()
        assert model_name.lower()in ['gru','lstm'],"check 'model_name' value in ['gru','lstm']"
        self.issplit = issplit
        self.channels = channels
        if model_name.lower()=='gru':
            self.rnn1 = nn.GRU(input_size=input_size,hidden_size=n_hid,num_layers=layer)
            if self.issplit:
                if isinstance(gap ,type(1)) and isinstance(time_step ,type(1)):
                    self.gap = gap
                    self.time_step = time_step
                    assert self.time_step % self.gap  == 0, "mske sure 'time_step' is divisible by 'gap'"
                    self.rnn2 = nn.GRU(n_hid,n_hid)
        else:
            self.rnn1 = nn.LSTM(input_size=input_size, hidden_size=n_hid, num_layers=layer)
            if self.issplit:
                if isinstance(gap, type(1)) and isinstance(time_step, type(1)):
                    self.gap = gap
                    self.time_step = time_step
                    assert self.time_step % self.gap == 0, "mske sure 'time_step' is divisible by 'gap'"
                    self.rnn2 = nn.LSTM(n_hid, n_hid)

    def forward(self, inputs):
        #input:(batch,chans,seq_len)
        x,_ = self.rnn1(inputs)
        if self.issplit:
            x = torch.stack([x[(i+1)*self.gap-1,:,:] for i in range(self.time_step//self.gap)],0)
            x, _ = self.rnn2(x) #(time_step//gap, batch*chans, num_hidden)
        return x

class RNN_CN(nn.Module):
    def __init__(self, config, gap,device,model_name ='gru'):
        super(RNN_CN, self).__init__()
        self.config = config
        assert self.config.sequence_length % self.config.window_size == 0, "mske sure 'sequence_length' is divisible by 'window_size'."
        assert isinstance(gap ,type(1))==1, "make sure 'gap' is int variable."
        assert self.config.sequence_length // self.config.window_size % gap == 0, "mske sure 'time_stepgap' is divisible by 'gap'"

        self.device = device
        self.gap = gap #control the group of sequence,if
        self.rel_rec = self._RS_construct(dim=1)
        self.rel_send = self._RS_construct(dim=0)

        self.rnn_Base_1 = rnn_base(self.config.window_size, self.config.n_hid, self.config.rnn_layers,
                                     self.config.channels, self.config.sequence_length // self.config.window_size,
                                     issplit=True,gap=gap,model_name=model_name)
        self.rnn_Base_2 = rnn_base(self.config.n_hid * 2, self.config.n_hid, self.config.rnn_layers,self.config.channels,model_name=model_name)
        self.rnn_Base_3 = rnn_base(self.config.n_hid, self.config.n_hid, self.config.rnn_layers, self.config.channels,model_name=model_name)
        self.rnn_Base_4 = rnn_base(self.config.n_hid * 2, self.config.n_hid, self.config.rnn_layers, self.config.channels,model_name=model_name)
        self.rnn_Base_5 = rnn_base(self.config.n_hid * 2, self.config.n_hid, self.config.rnn_layers, self.config.channels,model_name=model_name)
        self.block = nn.Sequential(
            MLP(self.config.n_hid,self.config.n_hid*2, self.config.n_hid*2),#old MLP(self.config.n_hid,self.config.n_hid, self.config.n_hid)
            MLP(self.config.n_hid*2, self.config.n_hid, self.config.n_hid), #old MLP(self.config.n_hid, self.config.n_hid, self.config.n_hid)
            MLP(self.config.n_hid, self.config.n_hid, 1),
            nn.ReLU()) 
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def edge2node(self, x, rel_rec, rel_send):
        # assumes that we have the same graph across all samples.(Neural Relational Inference for Interacting Systems,2018)
        incoming = torch.matmul(rel_rec.t(), x)
        return incoming / incoming.size(1)

    def node2edge(self, x, rel_rec, rel_send):
        # assumes that we have the same graph across all samples.(Neural Relational Inference for Interacting Systems,2018)
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)

        receivers = self._fun(receivers,receivers.size(-1)//self.config.n_hid)#(timestep,batch*dim,n_hid)
        senders = self._fun(senders,senders.size(-1)//self.config.n_hid)#(timestep,batch*dim,n_hid)
        # edges = torch.cat([senders,receivers],dim = -1)#(time_step, batch*dims,n_hid*2)
        edges = torch.cat([receivers, senders], dim=-1)
        return edges

    def _RS_construct(self,dim):
        #构造 R 和 S 矩阵，dim=1、0
        off_diag = np.ones([self.config.channels, self.config.channels])
        output = np.array(encode_onehot(np.where(off_diag)[dim]), dtype=np.float32)
        return torch.FloatTensor(output).to(self.device)

    def _fun(self,input,time_step):
        # input:(batch,dim, time_step*trans_dim)->#(time_step, batch*dim,trans_dim)
        input = input.view(-1,input.size(-1))#(batch*dim, time_step*trans_dim) 往后拼接
        input = torch.chunk(input,time_step,dim=-1)# a tuple conclude the element of (batch*dim,trans_dim)
        input = torch.stack(input,dim=0)#(time_step, batch*dim,trans_dim)
        return input

    def _fun_transpose(self,input,dim):
        # x:(time_step,batch*dim,n_hid)-> (batch,dim,time_step*n_hid)
        input = input.permute(1, 0, 2).contiguous()  # (batch*dim,time_step,n_hid)
        input = input.view(input.size(0), -1)  # (batch*dim,time_step*n_hid)
        input = input.view(-1, dim, input.size(-1))  # (batch,dim,time_step*n_hid)
        return input

    def forward(self, inputs):
        # Input shape: (batch , chans , seq_len)t5
        assert inputs.size(1) == self.config.channels,"check 'channels' variable in the Config"
        x = self._fun(inputs,self.config.sequence_length//self.config.window_size) #(time_step, batch*chans, window_size)

        x = self.rnn_Base_1(x) #(time_step, batch*chans, n_hid)
        x = self._fun_transpose(x,self.config.channels) #(batch,chans,time_step*n_hid)

        x = self.node2edge(x,self.rel_rec,self.rel_send)#(time_step, batch*chans*chans,n_hid*2)
        short_cut =x = self.rnn_Base_2(x) #(time_step, batch*chans*chans,n_hid)
        x = self._fun_transpose(x,dim = self.config.channels * self.config.channels )
        #(time_step, batch*chans*chans,n_hid)->(batch,chans*chans,time_step*n_hid)

        x = self.edge2node(x,self.rel_rec,self.rel_send)#(batch,chans,time_step*n_hid)
        x = self._fun(x,x.size(-1)//self.config.n_hid) #(time_step,batch*chans,n_hid)

        x =  self.rnn_Base_3(x)#(time_step,batch*chans,n_hid)
        x = self._fun_transpose(x, dim=self.config.channels) #(time_step,batch*chans,n_hid)-># (batch,chans,time_step*n_hid)

        x = self.node2edge(x,self.rel_rec,self.rel_send) #(time_step, batch*chans*chans,n_hid*2)
        x = self.rnn_Base_4(x)#(time_step, batch*chans*chans,n_hid)

        x = torch.cat([short_cut,x],dim = -1)#(time_step, batch*chans*chans,n_hid*2)
        x = self.rnn_Base_5(x)#(time_step, batch*chans*chans,n_hid)

        x = x[-1,:,:].unsqueeze(dim=0)#get last slice（1,batch*chans*chans,n_hid）
        x = self._fun_transpose(x, dim = self.config.channels * self.config.channels)#(batch,chans*chans,n_hid)

        x =self.block(x)
        x = x.reshape(-1, inputs.shape[1], inputs.shape[1])
        return x

if __name__ == '__main__':
    import config
    from torchinfo import summary 
    config = config.Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = RNN_CN(config, gap=4,device= device).to(device)
    print(summary(encoder, input_size=(2,5,512)))


