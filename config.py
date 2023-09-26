
class Config(object):
    num_works = 2  
    low_snr = -5
    high_snr = 10
    window_size = 32#32#size of sliding window
    gap = 4 #the number slice of sequence
    n_hid = 32 #
    rnn_layers = 2
    sequence_length = 1536 #512
    channels = 5 # channels of signal    val_radio = 0.2
    trainlnum = 20000# total_dataset, included train samples and val samples
    shuffle_dataset = True  # if shuffle dataset

    batch_size = 200#1500
    test_batch = 200#1500
    num_epochs = 800

    val_radio = 0.2
    learning_rate = 0.008   
    model_name = 'lstm' #'gru'#'lstm'
    assert model_name.lower()in ['gru','lstm'],"check 'model_name' value in ['gru','lstm']"

