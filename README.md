# A Spatio-Temporal Neural Relation Extraction Model for Mapping End-to-End Brain Directed Network 
In this work, we provides an idea on how to mining the interaction among long time series with supervised learning。Sepecially, we designed an deep learning model to directly establish the mapping from temporal signal to explicit causality graph instead of mapping for signal fitting. The model was trained with labeled data in a supervised way and can generalize universal causal inference knowledge, which means it can directly infer the potential directed relationships of new samples based on the learned experience.

## Introduction of training set
During the training process, the input and output training samples are as follows. In Figure (a), we use real signal amplitude information and multivariate autoregressive models to iteratively generate multi-dimensional time series signals, where each node corresponds to a one-dimensional time series signal. Figure (b) is the relationship interaction (label), a[i][j] represents the information flow relationship from node j to node i (the matrix is transposed during training). Figure (c) shows the directed connection pattern between nodes. The red line represents the bi-connection, and the connection with an arrow represents the flow of information from node j to node i.
![trainSample](https://github.com/WanjunCC/LSTM_GNN/blob/main/image/train_pair.png)


## Spatio-Temporal Neural Relation Extraction Model
The main model structure is presented as followed, which requires multivariate time series (MTS) input and outputs the directed network graph directly. It adopts an encoder-decoder framework and consists of two joint parts: an encoder equipped with temporal processing and GNN modules, and a share-weight ANN decoder.  The temporal processing module is designed to learn compelling and succinct sequence representations. The GNN module is equipped with multi-rounds of node-to-edge (v->e) and edge-to-node (e->v) message passing and updated by the multiple long short-term memory units (Multi-LSTMs) to obtain the spatial and temporal interaction representations. Finally, the decoder (a share-weight ANN) decodes all possible underlying interactions and scores the interacting effect. 

![Model](https://github.com/WanjunCC/LSTM_GNN/blob/main/image/model.png)

subfigure (a) presents the structure of the main model; below it, subfigures (b)-(c) show the detailes submodules in spatio-temporal neural relation extraction(STNRE), including the architecture of node to edge (b), edge to node (c), Multi-LSTMs (d) and LSTM (e). 

## Performance test (partial display)
