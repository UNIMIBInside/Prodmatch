import time
import torch
from torch import nn
from tqdm import tqdm
from pandas import pandas
from collections import OrderedDict
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from os import path
from ... import DATA_DIR, DEEPMATCH_DIR, RESULTS_DIR, CACHE_DIR


class Statistics(object):
    """Accumulator for loss statistics, inspired by ONMT.

    Keeps track of the following metrics:
    * F1
    * Precision
    * Recall
    * Accuracy
    """

    def __init__(self):
        self.loss_sum = 0
        self.examples = 0
        self.tps = 0
        self.tns = 0
        self.fps = 0
        self.fns = 0
        self.start_time = time.time()

    def update(self, loss=0, tps=0, tns=0, fps=0, fns=0):
        examples = tps + tns + fps + fns
        self.loss_sum += loss * examples
        self.tps += tps
        self.tns += tns
        self.fps += fps
        self.fns += fns
        self.examples += examples

    def loss(self):
        return self.loss_sum / self.examples

    def f1(self):
        prec = self.precision()
        recall = self.recall()
        return 2 * prec * recall / max(prec + recall, 1)

    def precision(self):
        return 100 * self.tps / max(self.tps + self.fps, 1)

    def recall(self):
        return 100 * self.tps / max(self.tps + self.fns, 1)

    def accuracy(self):
        return 100 * (self.tps + self.tns) / self.examples

    def examples_per_sec(self):
        return self.examples / (time.time() - self.start_time)


class Runner(object):

    @staticmethod
    def __compute_stats(output, target):
        # Get indices of max values per batch
        predictions = output.max(1)[1].data
        correct = (predictions == target.data).float()
        incorrect = (1 - correct).float()
        positives = (target.data == 1).float()
        negatives = (target.data == 0).float()

        tp = torch.dot(correct, positives)
        tn = torch.dot(correct, negatives)
        fp = torch.dot(incorrect, negatives)
        fn = torch.dot(incorrect, positives)

        return tp, tn, fp, fn

    @staticmethod
    def __print_final_stats(epoch, runtime, datatime, stats):
        """Write out epoch statistics to stdout.
        """
        print(('Finished Epoch {epoch} || Run Time: {runtime:6.1f} | '
               'Load Time: {datatime:6.1f} || F1: {f1:6.2f} | Prec: {prec:6.2f} | '
               'Rec: {rec:6.2f} | Accu: {accu:6.2f} || Ex/s: {eps:6.2f}\n').format(
                   epoch=epoch,
                   runtime=runtime,
                   datatime=datatime,
                   f1=stats.f1(),
                   prec=stats.precision(),
                   rec=stats.recall(),
                   accu=stats.accuracy(),
                   eps=stats.examples_per_sec()))

    @staticmethod
    def __set_pbar_status(pbar, stats, cum_stats):
        postfix_dict = OrderedDict([
            ('Loss', '{0:7.4f}'.format(stats.loss())),
            ('F1', '{0:7.2f}'.format(stats.f1())),
            ('Cum F1', '{0:7.2f}'.format(cum_stats.f1()))
        ])
        pbar.set_postfix(ordered_dict=postfix_dict)

    @staticmethod
    def __run(
            epoch,
            model,
            loader,
            criterion=None,
            optimizer=None,
            train=True,
            return_predictions=False,
            log_freq=4):

        datatime = 0
        runtime = 0
        cum_stats = Statistics()
        statistics = Statistics()
        predictions = []
        batch_end = time.time()

        with tqdm(total=len(loader.dataset)) as pbar:
            # Runner.__set_pbar_status(pbar, statistics, cum_stats)
            postfix_dict = OrderedDict([
                ('Loss', '{0:7.4f}'.format(00.0000)),
                ('F1', '{0:7.2f}'.format(00.00)),
                ('Cum F1', '{0:7.2f}'.format(00.00))
            ])
            pbar.set_postfix(ordered_dict=postfix_dict)

            if train:
                model.train()
                pbar.set_description('Train epoch: ' + str(epoch + 1))
            else:
                model.eval()
                pbar.set_description('Valid epoch: ' + str(epoch + 1))

            for batch_idx, (data, labels) in enumerate(loader):
                batch_start = time.time()
                datatime += batch_start - batch_end

                output = model(data)
                loss = float('NaN')
                if criterion:
                    loss = criterion(output, labels)
                stats = Runner.__compute_stats(output, labels)

                cum_stats.update(float(loss), *stats)
                statistics.update(float(loss), *stats)

                if return_predictions:
                    for probs in output:
                        # Get only the probability of being a match
                        predictions.append(float(probs[1].exp()))

                if (batch_idx + 1) % log_freq == 0:
                    Runner.__set_pbar_status(pbar, statistics, cum_stats)
                    statistics = Statistics()

                if train:
                    model.zero_grad()
                    loss.backward()
                    optimizer.step()

                batch_end = time.time()
                runtime += batch_end - batch_start
                pbar.update(len(data))

        Runner.__print_final_stats(epoch + 1, runtime, datatime, cum_stats)

        if return_predictions:
            return predictions
        else:
            return cum_stats

    @staticmethod
    def train(
            train_dataset,
            val_dataset,
            model,
            resume=False,
            criterion=None,
            optimizer=None,
            scheduler=None,
            train_epochs=10,
            pos_neg_ratio=1,
            best_model_name='best_model',
            best_save_on='F1',
            best_save_path=None,
            device=None,
            batch_size=32,
            **kwargs):

        if device is None:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        if criterion is None:
            if pos_neg_ratio >= 1:
                weight = torch.tensor([1/pos_neg_ratio, 1])
            else:
                weight = torch.tensor([1, 1/pos_neg_ratio])
            criterion = nn.NLLLoss(weight=weight)
        if optimizer is None:
            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        if scheduler is None:
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=0.1,
                patience=5,
                verbose=True)
        if best_save_path is None:
            best_save_path = path.join(RESULTS_DIR, 'models')
        if resume:
            print('Resuming best saved model')
            load = torch.load(
                path.join(best_save_path, best_model_name + '.pth'))
            train_epochs = load['train_epochs'] - load['epoch']
            model.load_state_dict(load['model_state_dict'])
            criterion.load_state_dict(load['criterion_state_dict'])
            optimizer.load_state_dict(load['optimizer_state_dict'])
            scheduler.load_state_dict(load['scheduler_state_dict'])
            print('Done')
        if best_save_on is None:
            best_save_on = 'F1'
        best_save_on = best_save_on.lower()
        assert(
            best_save_on == 'precision' or
            best_save_on == 'accuracy' or
            best_save_on == 'recall' or
            best_save_on == 'f1'
        )

        model.to(device)
        criterion.to(device)

        train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
        best = 0.0

        for epoch in range(train_epochs):
            Runner.__run(
                epoch,
                model,
                train_loader,
                criterion,
                optimizer,
                train=True,
                **kwargs
            )
            stats = Runner.__run(
                epoch,
                model,
                val_loader,
                criterion,
                optimizer,
                train=False,
                **kwargs
            )
            if getattr(stats, best_save_on)() > best:
                best = getattr(stats, best_save_on)()
                print('Saving best model')
                torch.save({
                    'epoch': epoch,
                    'train_epochs': train_epochs,
                    'model_state_dict': model.state_dict(),
                    'criterion_state_dict': criterion.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_' + best_save_on: getattr(stats, best_save_on)()
                }, path.join(best_save_path, best_model_name + '.pth'))
                print('Done')
            scheduler.step(getattr(stats, best_save_on)())
        return best

    @staticmethod
    def predict(
            dataset,
            model,
            load_best_model=True,
            best_model_name='best_model',
            best_save_path=None,
            device=None,
            batch_size=32,
            **kwargs):

        if device is None:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        if load_best_model:
            if best_save_path is None:
                best_save_path = path.join(
                    RESULTS_DIR, 'models', best_model_name + '.pth')
            print('Loading best model')
            load = torch.load(best_save_path)
            model.load_state_dict(load['model_state_dict'])
            print('Done')
        model.to(device)
        loader = DataLoader(dataset, batch_size, shuffle=False)
        predictions = Runner.__run(
            epoch=0,
            model=model,
            loader=loader,
            train=False,
            return_predictions=True,
            **kwargs
        )
        return predictions
