# 공통 부분
import sys, torch, random
import numpy as np
import matplotlib.pyplot as plt

class BPRDataset(torch.utils.data.Dataset):
    def __init__(self, user_item_pairs, num_items):
        self.pairs = user_item_pairs
        self.num_items = num_items
        self.user_pos = {}
        for u, i in self.pairs:
            self.user_pos.setdefault(u, set()).add(i)

    def __len__(self): 
        return len(self.pairs)

    def __getitem__(self, idx):
        u, i = self.pairs[idx]
        while True:
            j = random.randint(0, self.num_items - 1)
            if j not in self.user_pos[u]:
                return torch.LongTensor([u]), torch.LongTensor([i]), torch.LongTensor([j])

def bpr_loss(score, model, l2=1e-4):
    loss = -torch.mean(torch.log(torch.sigmoid(score)))
    reg = sum(torch.norm(p) for p in model.parameters())
    return loss + l2 * reg

def hit_at_k(ranked_list, ground_truth, k):
    return int(any(item in ranked_list[:k] for item in ground_truth))

def ndcg_at_k(ranked_list, ground_truth, k):
    dcg = 0.0
    for i, item in enumerate(ranked_list[:k]):
        if item in ground_truth:
            dcg += 1 / np.log2(i + 2)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(ground_truth), k)))
    return dcg / idcg if idcg > 0 else 0.0

def recall_at_k(ranked_list, ground_truth, k):
    hit = sum(1 for item in ground_truth if item in ranked_list[:k])
    return hit / len(ground_truth) if len(ground_truth) > 0 else 0.0

def setup_logger(log_path):
    class Logger(object):
        def __init__(self):
            self.terminal = sys.stdout
            self.log = open(log_path, "w")

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            self.terminal.flush()
            self.log.flush()

    sys.stdout = Logger()


def plot_metrics(loss_list, hit_list, ndcg_list, recall_list):
    epochs = list(range(1, len(loss_list)+1))
    val_epochs = list(range(5, len(loss_list)+1, 5))

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(epochs, loss_list, label="BPR Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")

    plt.subplot(2, 1, 2)
    plt.plot(val_epochs, hit_list, label="Hit@10")
    plt.plot(val_epochs, ndcg_list, label="nDCG@10")
    plt.plot(val_epochs, recall_list, label="Recall@10")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title("Validation Metrics")
    plt.legend()

    plt.tight_layout()
    plt.savefig("results/metrics_plot.png")
    plt.close()