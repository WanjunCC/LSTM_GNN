# LSTM_GNN
In this programme, we provides an idea on how to mining the interaction among long time series, and our work is under review in IEEE NNLS


During the training process, the input and output training samples are as follows. In Figure (a), we use real signal amplitude information and multivariate autoregressive models to iteratively generate multi-dimensional time series signals, where each node corresponds to a one-dimensional time series signal. Figure (b) is the relationship interaction (label), a[i][ j] represents the information flow relationship from node j to node i (the matrix is transposed during training). Figure (c) shows the directed connection pattern between nodes.
![trainSample](https://github.com/WanjunCC/LSTM_GNN/blob/main/image/train_pair.png)
