# A Spatio-Temporal Neural Relation Extraction Model for Mapping End-to-End Brain Directed Network 
In this work, we provides an idea on how to mining the interaction among long time series with supervised learning. Sepecially, we designed an deep learning model to directly establish the mapping from temporal signal to explicit causality graph instead of mapping for signal fitting. The model was trained with labeled data in a supervised way and can generalize universal causal inference knowledge, which means it can directly infer the potential directed relationships of new samples based on the learned experience.

## environment
- numpy shubld be <= 1.16.5
- torch	1.9.0	
- torchaudio	0.9.0	
- torchvision	0.10.0
- python 3.7

## Introduction of training set
During the training process, the input and output training samples are as follows. In Figure (a), we use real signal amplitude information and multivariate autoregressive models to iteratively generate multi-dimensional time series signals, where each node corresponds to a one-dimensional time series signal. Figure (b) is the relationship interaction (label), a[i][j] represents the information flow relationship from node j to node i (the matrix is transposed during training). Figure (c) shows the directed connection pattern between nodes. The red line represents the bi-connection, and the connection with an arrow represents the flow of information from node j to node i.
![trainSample](https://github.com/WanjunCC/LSTM_GNN/blob/main/image/train_pair.png)


## Spatio-Temporal Neural Relation Extraction Model
The main model structure is presented as followed, which requires multivariate time series (MTS) input and outputs the directed network graph directly. It adopts an encoder-decoder framework and consists of two joint parts: an encoder equipped with temporal processing and GNN modules, and a share-weight ANN decoder.  The temporal processing module is designed to learn compelling and succinct sequence representations. The GNN module is equipped with multi-rounds of node-to-edge (v->e) and edge-to-node (e->v) message passing and updated by the multiple long short-term memory units (Multi-LSTMs) to obtain the spatial and temporal interaction representations. Finally, the decoder (a share-weight ANN) decodes all possible underlying interactions and scores the interacting effect. 

![Model](https://github.com/WanjunCC/LSTM_GNN/blob/main/image/model.png)

subfigure (a) presents the structure of the main model; below it, subfigures (b)-(c) show the detailes submodules in spatio-temporal neural relation extraction(STNRE), including the architecture of node to edge (b), edge to node (c), Multi-LSTMs (d) and LSTM (e). 

## Performance test (partial display)
Comparison using a wide range of tools: Granger causality analysis
### Simulation results
Recovery of predefined network patterns with 50 runs at different SNRs, where the red line represents the main network patterns captured by both models (the first is ours). Line thickness represents the total numbers this connection appears in the 50 runs of simulation, and the distinct color indicates the predominance of a connection, i.e., the line appears more (red) or less (gray) than μ+σ times in the 50 runs of simulations. 

![recovery](https://github.com/WanjunCC/LSTM_GNN/blob/main/image/result1.png)


Our model can be generalized in the causal inference for sequences of different lengths or different numbers of node. We compare the time costs of the two methods under different input forms (sequence length, number of nodes), and our method has the advantage in time cost(All the tests were conducted on a CPU device: Inter(R) Core (TM) i7-8750H CPU@2.20GHz). 

![Timecost](https://github.com/WanjunCC/LSTM_GNN/blob/main/image/result2.png)

### Real Signal test
We collected resting state electroencephalogram(EEGs) from more than a dozen normal people for 5 minutes. After preprocessing, the signal is cut into lengths of 2-, 4-, 6-, 8s and input into the model to build the brain network.

![nework](https://github.com/WanjunCC/LSTM_GNN/blob/main/image/result3.png)


Network consistency under different epoch lengths(mean ± standard deviation). And a star denotes the consistency of DeepNNetDNE is significantly higher (paired t-test, one tail, p<0.00).

![network](https://github.com/WanjunCC/LSTM_GNN/blob/main/image/result4.png)


Under different epoch lengths, both DeepNNetDNE and GCA show relatively consistent topologies that mainly involve connections symmetrically distributed in frontal, posterior parietooccipital, and posterior temporal regions. However, comparatively, the network estimated by DeepNNetDNE has higher consistency among different epoch lengths than that of GCA. 
As shown, DeepNNetDNE keeps relatively high consistency under different epoch lengths (higher than 0.57±0.05); and most of the time, is significantly higher than that of GCA (paired t-test, one tail, p<0.000). For GCA, the consistency under different epoch lengths is relatively lower, especially between the short (2 s or 4 s) and long epoch lengths (6 s or 8 s), e.g., 2 s vs. 8 s: 0.310±0.071. Only under the long epoch lengths, the GCA obtained higher consistency than DeepNNetDNE (i.e., the highest consistency of GCA, 6 s vs. 8 s: 0.822±0.060) but is lower than the highest consistency of DeepNNetDNE (4 s vs. 6 s: 0.913±0.028).

More test result are presented in our paper！
