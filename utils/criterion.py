from re import A
import torch
import numpy as np
from copy import deepcopy


class TaoBaoMetric(object):
    def __init__(self) -> None:
        super(TaoBaoMetric, self).__init__()
        self.n_images = 0
        self.n_correct_pairs = 0
        self.n_products = 0
        self.n_correct_products = 0
        self.loss = 0

    def init(self):
        self.n_images = 0
        self.n_correct_pairs = 0
        self.n_products = 0
        self.n_correct_products = 0
        self.loss = 0

    def get_value(self):
        ACC = (self.n_correct_pairs / self.n_images)
        EM = (self.n_correct_products / self.n_products)
        return ACC, EM

    def get_loss(self):
        return self.loss / self.n_images

    def get_acc(self):
        return (self.n_correct_pairs / self.n_images)

    def get_em(self):
        return (self.n_correct_products / self.n_products)

    def print_value(self):
        ACC, EM = self.get_value()
        print(ACC, EM)
        print('Accuracy: %.6f, Exact Match: %.6f' % (ACC, EM))

    def update(self, gt_labels, pred_labels):
        n_new_images = len(gt_labels)
        n_new_correct_pairs = np.sum(np.array(gt_labels) == np.array(pred_labels))
        self.n_images = self.n_images + n_new_images
        self.n_correct_pairs = self.n_correct_pairs + n_new_correct_pairs
        self.n_products = self.n_products + 1
        if n_new_correct_pairs == n_new_images:
            self.n_correct_products = self.n_correct_products + 1

    def increase_loss(self, loss):
        self.loss = self.loss + loss

if __name__ == '__main__':
    a = 15
    for i in range(10):
        a = a + 1
        print('a => ', a)
    print('a => ', a)
