import numpy as np

from model.config import cfg
import os
import csv
import copy

class StatCollector(object):

    def __init__(self, nbr_ep, stat_strings, ma_length=10, output_dir="", is_training=True):

        # Set mode
        self.is_training = is_training
        if self.is_training:
            filename = "training_stats.csv"
            self.filename = os.path.join(output_dir, filename)
        # What to display
        self.stat_strings = stat_strings
        self.nbr_stats = len(stat_strings)
        max_string_len = 0
        for i in range(self.nbr_stats):
            max_string_len = max(max_string_len, len(self.stat_strings[i]))
        self.spaces = []  # cell(self.nbr_stats, 1)
        for i in range(self.nbr_stats):  # = 1 : self.nbr_stats
            self.spaces.append((max_string_len - len(self.stat_strings[i])) * ' ')

        # saving loss, stats, else
        self.stats_loss = []
        self.stats_curr_batch_ce_done_rew_prod = []
        self.stats_curr_batch_ce_fix_rew_prod = []

        self.stats_data = {stat: [] for stat in stat_strings}
        self.stats_else = []

        # Initialize total averages
        self.mean_loss = 0.0
        self.mean_curr_batch_ce_done_rew_prod = 0.0
        self.mean_curr_batch_ce_fix_rew_prod = 0.0

        self.means = np.zeros(self.nbr_stats, dtype=np.float32)
        self.ma_factor = ma_length
        # Initialize exponential moving averages
        self.ma_loss = 0.0
        self.ma_curr_batch_ce_done_rew_prod = 0.0
        self.ma_curr_batch_ce_fix_rew_prod = 0.0

        self.mas = np.zeros(self.nbr_stats, dtype=np.float32)

        self.ma_weight = cfg.LRP_HAI_TRAIN.MA_WEIGHT
        self.nbr_ep = nbr_ep
        self.ep = 0
        self.bz = cfg.LRP_HAI_TRAIN.BATCH_SIZE

        if self.is_training:
            with open(self.filename, 'w', newline="") as csvfile:
                writer = csv.writer(csvfile)
                strings = copy.deepcopy(self.stat_strings)
                strings.insert(0, "iter")
                strings.insert(1, "loss")
                strings.insert(2, "ce_done_rew_prod")
                strings.insert(3, "ce_fix_rew_prod")
                writer.writerow(strings)

    def update(self, loss, curr_batch_ce_done_rew_prod, curr_batch_ce_fix_rew_prod, other):
        # Updates averages
        self.update_loss_stat_data(loss, curr_batch_ce_done_rew_prod, curr_batch_ce_fix_rew_prod, other)
        self.update_loss(loss)
        self.update_curr_batch_ce_done_rew_prod(curr_batch_ce_done_rew_prod)
        self.update_curr_batch_ce_fix_rew_prod(curr_batch_ce_fix_rew_prod)

        self.update_means_mas(other)
        self.ep += 1

    def update_loss_stat_data(self, loss, curr_batch_ce_done_rew_prod, curr_batch_ce_fix_rew_prod, other):
        else_len = len(other) - self.nbr_stats
        if (self.ep + 1) % self.bz == 0:
            self.stats_loss.append(loss)
            self.stats_curr_batch_ce_done_rew_prod.append(curr_batch_ce_done_rew_prod)
            self.stats_curr_batch_ce_fix_rew_prod.append(curr_batch_ce_fix_rew_prod)

        for i in range(self.nbr_stats):
            self.stats_data[self.stat_strings[i]].append(other[i])
        if else_len == 1:
            self.stats_else.append(other[-1])

    def update_loss(self, loss):
        # Tracks the loss
        if (self.ep + 1) % self.bz != 0:
            return
        batch_idx = (self.ep + 1) // self.bz - 1
        self.mean_loss = (batch_idx * self.mean_loss + loss) / (batch_idx + 1)
        # self.ma_loss = (1 - self.ma_weight) * self.ma_loss + self.ma_weight * loss
        self.ma_loss = np.mean(self.stats_loss[-1])

    def update_curr_batch_ce_done_rew_prod(self, curr_batch_ce_done_rew_prod):
        # Tracks the loss
        if (self.ep + 1) % self.bz != 0:
            return
        batch_idx = (self.ep + 1) // self.bz - 1
        self.mean_curr_batch_ce_done_rew_prod = (batch_idx * self.mean_curr_batch_ce_done_rew_prod + curr_batch_ce_done_rew_prod) / (batch_idx + 1)
        # self.ma_loss = (1 - self.ma_weight) * self.ma_loss + self.ma_weight * loss
        self.ma_curr_batch_ce_done_rew_prod = np.mean(self.stats_curr_batch_ce_done_rew_prod[-1])

    def update_curr_batch_ce_fix_rew_prod(self, curr_batch_ce_fix_rew_prod):
        # Tracks the loss
        if (self.ep + 1) % self.bz != 0:
            return
        batch_idx = (self.ep + 1) // self.bz - 1
        self.mean_curr_batch_ce_fix_rew_prod = (batch_idx * self.mean_curr_batch_ce_fix_rew_prod + curr_batch_ce_fix_rew_prod) / (batch_idx + 1)
        # self.ma_loss = (1 - self.ma_weight) * self.ma_loss + self.ma_weight * loss
        self.ma_curr_batch_ce_fix_rew_prod = np.mean(self.stats_curr_batch_ce_fix_rew_prod[-1])

    def update_means_mas(self, data):
        # Tracks various statistics
        for i in range(len(data)):
            if not isinstance(data[i], list):
                self.means[i] = (self.ep * self.means[i] + data[i]) / (self.ep + 1)
                self.mas[i] = np.mean(self.stats_data[self.stat_strings[i]][-self.ma_factor:])

    def print_stats(self, iter, logger=None):
        if logger:
            if self.is_training:
                logger.info('Mean loss (tot, MA):      (%f, %f)' % (self.mean_loss, self.ma_loss))
                logger.info('mean_curr_batch_ce_done_rew_prod (tot, MA):      (%f, %f)' % (self.mean_curr_batch_ce_done_rew_prod, self.ma_curr_batch_ce_done_rew_prod))
                logger.info('mean_curr_batch_ce_fix_rew_prod (tot, MA):      (%f, %f)' % (self.mean_curr_batch_ce_fix_rew_prod, self.ma_curr_batch_ce_fix_rew_prod))

            for i in range(self.nbr_stats):
                logger.info('Mean %s (tot, MA): %s(%f, %f)' \
                      % (self.stat_strings[i], self.spaces[i], self.means[i], self.mas[i]))
        else:
            if self.is_training:
                print('Mean loss (tot, MA):      (%f, %f)' % (self.mean_loss, self.ma_loss))
            for i in range(self.nbr_stats):
                print('Mean %s (tot, MA): %s(%f, %f)' \
                      % (self.stat_strings[i], self.spaces[i], self.means[i], self.mas[i]))

        if self.is_training:
            with open(self.filename, 'a', newline="") as csvfile:
                writer = csv.writer(csvfile)
                row = []
                row.append(iter)
                row.append(self.ma_loss)
                row.append(self.ma_curr_batch_ce_done_rew_prod)
                row.append(self.ma_curr_batch_ce_fix_rew_prod)

                for i in self.mas:
                    row.append(i)
                writer.writerow(row)
