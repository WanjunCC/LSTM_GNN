import numpy as np
from datagenerator import *
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T
from model.module import *
import matplotlib.pyplot as plt
import scipy.io as scio
import torch
from config import *
from torch.autograd import Variable

# numpy shubld be <= 1.16.5
def calculate_confusion(A_True, A_predict, return_flag=False):
    """Diagonal elements are not considered"""
    assert A_True.shape == A_predict.shape
    dig_number = A_True.shape[-1]
    for i in range(dig_number):  # 对角线置0值
        A_True[:, i, i] = 0
        A_predict[:, i, i] = 0
    # threshold
    mean_T = np.expand_dims(np.sum(A_True.reshape((-1, A_True.shape[-1]*A_True.shape[-2])), axis=-1) / (dig_number * (dig_number - 1)),axis=-1)  # (1000,1)
    A_True = np.where(A_True.reshape((-1, A_True.shape[-1]*A_True.shape[-2])) > mean_T, 1, 0)  # （？，channel*channel）
    edge_num = np.sum(A_True, axis=-1)

    # Remove the all-0 matrix
    k = np.sum(A_True, axis=-1)
    error_index = np.where(k == 0)
    A_True = np.delete(A_True, error_index, axis=0).reshape((-1, dig_number * dig_number))
    A_predict = np.delete(A_predict, error_index, axis=0).reshape((-1, dig_number * dig_number))  # (1000,25)
    edge_num = np.delete(edge_num, error_index, axis=0)

    # A_predict
    K = np.sort(A_predict, axis=-1)  # sort from smallest to largest
    index = [(i, -edge_num[i]) for i in range(len(edge_num))]
    row, col = zip(*index)
    thousld = np.expand_dims(K[row, col], axis=-1)  # (1000,1)
    A_predict = np.where(A_predict >= thousld, 1, 0)

    # Recall= TP/ (TP+FN)
    TP = np.sum(np.logical_and(A_True, A_predict), axis=-1)
    total_connect = dig_number * (dig_number - 1)
    Recall = TP / np.sum(A_True.reshape((-1, dig_number * dig_number)), axis=-1)

    Accuracy = (total_connect - np.sum(np.logical_xor(A_True, A_predict).reshape((-1, dig_number * dig_number)),
                                       axis=-1)) / total_connect
    precision = TP / np.sum(A_predict.reshape((-1, dig_number * dig_number)), axis=-1)

    precision[np.isnan(precision)] = 0

    F1_score = 2 * precision * Recall / (precision + Recall)
    F1_score[np.isnan(F1_score)] = 0
    if return_flag:
        return [Recall, Accuracy, precision, F1_score], [A_True.reshape((-1,dig_number,dig_number)), A_predict.reshape((-1,dig_number,dig_number))]
    return [Recall, Accuracy, precision, F1_score], None

class Network(object):
    def __init__(self, config, model_dir):
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.Model = self.build()
        self.model_dir = model_dir  # saved dir of model weight  

    def build(self):
        model = RNN_CN(self.config,gap=self.config.gap,device= self.device, model_name =self.config.model_name)
        return model

    def find_last(self, weight_dir):
        """Finds the last checkpoint file of the last trained model in the
                model directory.
        Returns:
            The path of the last weight file
        """
        file_names = next(os.walk(weight_dir))[2]
        key = self.config.model_name
        file_names = filter(lambda f: f.startswith(key), file_names)
        file_num = [int(i.split("_")[-1].split(".")[0]) for i in file_names]
        file_num = sorted(file_num) 
        if not file_names:
            import errno
            raise FileNotFoundError(
                errno.ENOENT,
                "Could not find weight files in {}".format(weight_dir))
        self.epoch = file_num[-1] + 1
        file_name = os.path.join(weight_dir, key + "_" + str(file_num[-1]) + ".pt")
        return file_name

    def train(self, data_dir, mode, init_with="None", visualize="tensorboard", weight_path=None, random_split=True):
        """
        :param
        'init_with':
            'None':retrain the model new;
            'last':find the last weight to train;
            'specific':find the specific weight file to train.
        :return:
        """
        assert visualize in ["tensorboard", "matlab"], "make sure 'visualize' in 'tensorboard' or 'matlab'!"
        assert init_with in ["None", "last", "specific"], "make sure 'init_with' in 'None' or 'last' or 'specific'!"
        assert mode in ["training", "inference"], "make sure 'mode' in 'training' or 'inference'"

        # split data and data generators
        transform = T.Compose([addRandom_gnoise(self.config.low_snr, self.config.high_snr)])
        dataset = Signalset(data_dir=data_dir, transform=transform, flag=mode)

        ###############         random split the dataset          ##########################
        indices = list(range(self.config.trainlnum))
        split = int(np.floor(self.config.val_radio * self.config.trainlnum))
        if self.config.shuffle_dataset:
            np.random.seed(0)
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)

        train_loader = DataLoader(dataset, batch_size=self.config.batch_size, sampler=train_sampler,
                                  num_workers=self.config.num_works)
        val_loader = DataLoader(dataset, batch_size=self.config.test_batch, sampler=val_sampler,
                                num_workers=self.config.num_works)
        loader = {'train': train_loader, 'val': val_loader}

        # create model_dir if it does not exist
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # Setting
        self.epoch = 1

        optimizer = torch.optim.Adam(params=self.Model.parameters(), lr=self.config.learning_rate)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 400 ,600], gamma=0.8) # MultiStepLR
        criterion = Myloss(sqrting=False)
        # adjustment of learning rate
        if init_with in ["last", "specific"]:
            if os.path.exists(weight_path):
                if init_with == "last":
                    weight_path = self.find_last(weight_path)
                check_point = torch.load(weight_path, map_location=self.device)
                self.Model.load_state_dict(check_point['model'])
                optimizer.load_state_dict(check_point['optimizer'])
                scheduler.load_state_dict(check_point["scheduler"])
                if torch.cuda.is_available():
                    for state in optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.cuda()
            else:
                import errno
                raise FileNotFoundError(
                    errno.ENOENT,
                    "Could not find weight files in {}".format(weight_path))
        if visualize == "tensorboard":
            writer = SummaryWriter(self.model_dir)
        # Train
        self.Model.to(self.device)
        print("\nStarting at epoch {}. LR={}".format(self.epoch, self.config.learning_rate))

        min_loss = 10
        assert self.epoch < self.config.num_epochs, "Please check the num_epochs set"
        for epoch in range(self.epoch, self.config.num_epochs + 1):
            for moding in ['train', 'val']:
                losses = []
                if moding == "train":
                    self.Model.train = True
                    print('\nEpoch {}/{}\t learn_rate:{}'.format(epoch, self.config.num_epochs,
                                                                 optimizer.state_dict()['param_groups'][0]['lr']))
                else:
                    self.Model.train = False
                tot_loss = 0.0  # 每个batch置为0
                num = 0
                for data in loader[moding]:
                    singals, labels = data
                    if torch.cuda.is_available():
                        singals, labels = Variable(singals.cuda()), Variable(labels.cuda())
                    else:
                        singals, labels = Variable(singals), Variable(labels)
                    if moding == "train":
                        optimizer.zero_grad()
                    A = self.Model(singals[:,:,:self.config.sequence_length]).to(torch.device("cpu"))  # 取出来放入CPU计算

                    loss = criterion(A, labels.to(torch.device("cpu")))
                    losses.append(loss.item())
                    if moding == 'train':
                        loss.backward()
                        optimizer.step()
                    tot_loss += loss.data
                    num += 1
                if moding == "train":
                    scheduler.step()
                    epoch_loss = tot_loss / num
                    lr = optimizer.state_dict()['param_groups'][0]['lr']
                    print('train loss:{}'.format(epoch_loss))
                    if visualize == "tensorboard":
                        writer.add_scalar('Loss_train', epoch_loss, epoch)
                        writer.add_scalar('lr', lr, epoch)
                else:
                    epoch_loss = tot_loss / num
                    print('val loss:{}'.format(epoch_loss))
                    if visualize == "tensorboard":
                        writer.add_scalar('Loss_val', epoch_loss, epoch)
                    if epoch_loss < min_loss:
                        min_loss = epoch_loss
                        state = {'model': self.Model.state_dict(), 'optimizer': optimizer.state_dict(),
                                 'scheduler': scheduler.state_dict()}
                        torch.save(state,
                                   os.path.join(self.model_dir, self.config.model_name + "_" + str(epoch) + ".pt"))

    def predict(self, weight_path, data = None, data_dir = None,device="cpu",is_written=False, written_dir=None):
        """
        'weight_path': the parameters saved path
        'data':if input is the matrix of data, set values, or is None
        'data_dir': if input is the dir of the data, set values, or is None
        'is_written':True: save the results; 'False': without saving
        """
        device = device.lower()
        assert device in ['cpu', 'gpu']
        if os.path.exists(weight_path):
            if device == "gpu":
                device = torch.device("cuda:0")
            else:
                device = torch.device(device)
            try:
                check_point = torch.load(weight_path, map_location=self.device)
                self.Model.load_state_dict(check_point['model'])
            except KeyError:
                self.Model.load_state_dict(torch.load(weight_path, map_location=device))
            self.Model.to(device)
            self.Model.eval()  # 测试模式

            # use "path_dir" to load data, data_dir is not equel to 'None'
            if  data is None and data_dir is not None:
                dataset = Signalset(data_dir=data_dir, flag="inference")
                loader = DataLoader(dataset, batch_size=self.config.test_batch, num_workers=self.config.num_works)
                with torch.no_grad():
                    output = []
                    for singals in loader:
                        singals = singals.to(device)
                        singals = (singals - torch.mean(singals, dim=-1).unsqueeze(dim=-1)) / \
                                  torch.std(singals, dim=-1).unsqueeze(dim=-1)
                        output.append(self.Model(singals).cpu().detach().numpy())
            else:
                with torch.no_grad():
                    output = []
                    for i in range(data.shape[0]//self.config.test_batch+1): 
                        try:
                            singals = torch.tensor(data[self.config.test_batch * i:self.config.test_batch * (i + 1), :,:],
                                                   dtype=torch.float)
                            singals = singals.to(device)
                            # normalization
                            singals = (singals - torch.mean(singals, dim=-1).unsqueeze(dim=-1)) / \
                                      torch.std(singals, dim=-1).unsqueeze(dim=-1)
                            output.append(self.Model(singals).cpu().detach().numpy())
                        except IndexError:
                            singals = torch.tensor(data[self.config.test_batch*i:, :, :], dtype=torch.float)
                            singals = singals.to(device)
                            # normalization
                            singals = (singals - torch.mean(singals, dim=-1).unsqueeze(dim=-1)) / \
                                      torch.std(singals, dim=-1).unsqueeze(dim=-1)
                            output.append(self.Model(singals).cpu().detach().numpy())
            output = np.vstack(output)
            output = output.reshape((-1,) + output.shape[-2:])

            if is_written and written_dir is not None:
                if not os.path.exists(written_dir):
                    os.makedirs(written_dir)
                for i in range(0, len(output)):
                    scio.savemat(os.path.join(written_dir, "dat" + str(i + 1) + ".mat"), {'dat': output[i, :, :]})
            return output
        else:
            import errno
            raise FileNotFoundError(
                errno.ENOENT,
                "{} is not avaliable".format(weight_path))

    def evaluate(self, data_dir, weight_path, device="cpu", is_written=False, written_dir=None, return_flag=False):
        """
        output included the prediction and the label
        'is_written': True,will save in '.mat' file
        """
        device = device.lower()
        assert device in ['cpu', 'gpu']
        if os.path.exists(weight_path):
            if device == "gpu":
                device = torch.device("cuda:0")
            else:
                device = torch.device(device)
            try:
                check_point = torch.load(weight_path, map_location=self.device)
                self.Model.load_state_dict(check_point['model'])
            except KeyError:
                self.Model.load_state_dict(torch.load(weight_path, map_location=device))
            self.Model.to(device)
            self.Model.eval() 

            dataset = Signalset(data_dir=data_dir, flag="training")
            loader = DataLoader(dataset, batch_size=self.config.test_batch, num_workers=self.config.num_works,shuffle=False)
            T_labels = []
            with torch.no_grad():
                output = []
                for singals, labels in loader:
                    singals = singals.to(device) 
                    # normalization
                    singals = (singals - torch.mean(singals, dim=-1).unsqueeze(dim=-1)) / \
                              torch.std(singals,dim=-1).unsqueeze(dim=-1)
                    output.append(self.Model(singals).cpu().detach().numpy())
                    T_labels.append(labels.cpu().detach().numpy())
            output = np.vstack(output)
            output = output.reshape((-1,) + output.shape[-2:])

            T_labels = np.vstack(T_labels)
            T_labels = T_labels.reshape((-1,) + output.shape[-2:])
            if return_flag:
                result, A = calculate_confusion(T_labels, output, return_flag)
            else:
                result, A = calculate_confusion(T_labels, output)
            # save the results
            if is_written and written_dir is not None:
                if not os.path.exists(written_dir):
                    os.makedirs(written_dir)
                for i in range(0, len(T_labels)):
                    scio.savemat(os.path.join(written_dir, "dat" + str(i + 1) + ".mat"), {'dat': output[i, :, :]})
            if return_flag:
                return output, T_labels, result, A
            else:
                return output, T_labels, result
        else:
            import errno
            raise FileNotFoundError(
                errno.ENOENT,
                "{} is not avaliable".format(weight_path))

if __name__ == '__main__':
    root_dir = os.getcwd()
    data_dir = r" "
    config = Config()
    work = Network(config=Config, model_dir=os.path.join(root_dir, "Model_64_1k5_1k5_10W_wsize_64_myloss_lstm2_8s"))

    ## the sample that train with the last weight file
    # work.train(init_with="last", mode="training", data_dir=data_dir,
    #            visualize="tensorboard", weight_path=os.path.join(root_dir,"Model_64_1k5_1k5_10W_wsize_64_myloss_lstm2_8s"))

    ### the sample that retrain
    work.train(init_with="None", mode="training", data_dir=data_dir,visualize="tensorboard", weight_path=None)

    ## the sample that predict
    # output, T_labels, result = work.evaluate(data_dir=data_dir,
    #                                          weight_path=os.path.join(root_dir, r"Gnn_64_4K_40W_MSEloss", "GNN_467.pt"),
    #                                          device='gpu', is_written=False, written_dir=r"Te st\test_pure")
    # Recall, Accuracy, precision, F1_score = result
    # print(Recall, Accuracy, precision, F1_score)
    print("ok")
