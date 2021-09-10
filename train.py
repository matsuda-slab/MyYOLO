import torch
import torch.nn as nn
import torch.optim as optim
from utils.datasets import _create_data_loader
from torch.utils.data import DataLoader
from utils.loss import compute_loss
from utils.utils import plot_graph
from model import YOLO, load_model
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--weights')
parser.add_argument('--batch_size', default=64)
parser.add_argument('--epochs', default=100)
parser.add_argument('--lr', default=0.0001)
parser.add_argument('--data_root', default='/home/users/matsuda/work/NN/yolov3/eriklindernoren/PyTorch-YOLOv3/data/coco/')
args = parser.parse_args()

DATA_ROOT    = args.data_root
BATCH_SIZE   = args.batch_size
EPOCHS       = args.epochs
LR           = args.lr
TRAIN_PATH   = DATA_ROOT + '/trainvalno5k.txt'
DECAY        = 0.0005
SUBDIVISION  = 2
BURN_IN      = 1000
lr_steps     = [[400000, 0.1], [450000, 0.1]]
weights_path = args.weights

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# データの準備・前処理
#    dataset = MyDataset(DATA_ROOT)
#    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Load training dataloader
dataloader = _create_data_loader(
    TRAIN_PATH,
    BATCH_SIZE,
    416
    )

# Load validation dataloader
# validation_dataloader = _create_validation_data_loader(
#     valid_path,
#     mini_batch_size,
#     model.hyperparams['height'],
#     args.n_cpu)

# モデルの生成
# model = YOLO().to(device)
model = load_model(weights_path, device)

# 最適化アルゴリズム, 損失関数の定義
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=DECAY)  # configに合わせて
# criterion = # loss算出クラス・関数を定義する

# lossのグラフ用リスト
losses = []

# 学習ループ
model.train()
print("Start Training\n")
start = time.ctime()
start_cnv = time.strptime(start)
for epoch in range(EPOCHS):
    for ite, (_, image, target) in enumerate(dataloader):
        batches_done = len(dataloader) * epoch + ite

        image = image.to(device)
        target = target.to(device)
        # forward
        outputs = model(image)

        # 損失の計算
        loss, _ = compute_loss(outputs, target, model)

        # backward
        loss.backward()

        ## 学習率の調整 及び optimizerの実行
        if batches_done % SUBDIVISION == 0:         # SUBDIVISION = 2なら, 2回に1回学習率が変わる
            lr = LR
            if batches_done < BURN_IN:
                lr *= (batches_done / BURN_IN)
            else:
                for threshold, value in lr_steps:
                    if batches_done > threshold:
                        lr *= value

            for g in optimizer.param_groups:
                g['lr'] = lr

            # optimizer を動作させる
            optimizer.step()

            optimizer.zero_grad()

        if batches_done % 10 == 0:
            print("[%3d][%d] Epoch / [%4d][%d] : loss = %.4f" % (epoch, EPOCHS, ite, len(dataloader), loss))



    losses.append(loss.item())

end = time.ctime()
end_cnv = time.strptime(end)

print("Start date >", time.strftime("%Y/%m/%d %H:%M:%S", start_cnv))
print("End date >", time.strftime("%Y/%m/%d %H:%M:%S", end_cnv))

# 学習結果(重みパラメータ)の保存
torch.save(model.state_dict(), "tiny-yolo.model")

# lossグラフの作成
plot_graph(losses, EPOCHS)
