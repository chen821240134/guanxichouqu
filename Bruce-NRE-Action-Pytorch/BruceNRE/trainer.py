
import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

def train(epoch, device, data_loader,model, optimizer, loss_fn , config):
    """
    :param epoch:
    :param device:
    :param data_loader:
    :param model:
    :param optimizer:
    :param loss:
    :param cofig:
    :return:
    """
    model.train()
    total_loss = []
    #model.train()是随机取一部分网络连接来训练更新参数，而model.eval()是利用到了所有网络连接。
    #如果模型中有BN层(Batch Normalization）和Dropout，在测试时添加model.eval()。model.eval()是保证BN层能够用全部训练数据的均值和方差，即测试过程中要保证BN层的均值和方差不变。对于Dropout，model.eval()是利用到了所有网络连接，即不进行随机舍弃神经元。

    for batch_idx, batch in enumerate(data_loader):
        *x, y = [data.to(device) for data in batch]#*x代表了四个数值
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        total_loss.append(loss)

        #记录日志
        data_cal = len(data_loader.dataset) if batch_idx == len(data_loader) else batch_idx * len(y)

        if(config.train_log and batch_idx % config.log_interval == 0) or batch_idx == len(data_loader):
            print('Traing epoch:{}[{}/{}({:.0f}%)]\tLoss:{:.6f}'.format(epoch,
                                                                         data_cal,
                                                                         len(data_loader.dataset),
                                                                        100.*batch_idx / len(data_loader),
                                                                        loss.item()
                                                                         ))
def validate(data_loader, model, device, config):
    """
    :param data_loader:
    :param model:
    :param device:
    :param config:
    :return:
    """
    model.eval()

    with torch.no_grad():
        total_y_true = np.empty(0)
        total_y_pred = np.empty(0)
        for batch_idx, batch in enumerate(data_loader):
            *x, y = [ data.to(device) for data in batch]
            y_pred = model(x)#128 10

            y_pred = y_pred.argmax(dim = -1)#128

            try:
                y, y_pred = y.numpy(), y_pred.numpy()
            except:
                y, y_pred = y.cpu().numpy(), y_pred.cpu().numpy()

            total_y_true = np.append(total_y_true, y)
            total_y_pred = np.append(total_y_pred, y_pred)

        total_f1 = []

        for average in config.f1_norm:#f1_norm = ['macro', 'micro']
            p, r, f1,_ = precision_recall_fscore_support(total_y_true, total_y_pred, average=average)

            print(f'{average} metrics:[p：{p:.4f}, r:{r:.4f}, f1:{f1:.4f}]')

            total_f1.append(f1)

    return total_f1