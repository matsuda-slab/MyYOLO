from utils.datasets import _create_data_loader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.loss import compute_loss
from utils.utils import plot_graph
from model import YOLO
import time

BATCH_SIZE = 64
EPOCHS     = 2
DATA_ROOT  = '/home/matsuda/datasets/COCO'
LR         = 0.0001
DECAY      = 0.0005

TRAIN_PATH = DATA_ROOT + '/2014/trainvalno5k.txt'

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
model = YOLO().to(device)

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
        image = image.to(device)
        target = target.to(device)
        # forward
        outputs = model(image)

        # 損失の計算
        loss, _ = compute_loss(outputs, target, model)
        print("[%3d][%d] Epoch / [%4d][%d] : loss = %.4f" % (epoch, EPOCHS, ite, len(dataloader), loss))

        # backward
        optimizer.zero_grad()
        loss.backward()

        # optimizer を動作させる
        optimizer.step()

    losses.append(loss.item())

end = time.ctime()
end_cnv = time.strptime(end)

print("Start date >", time.strftime("%Y/%m/%d %H:%M:%S", start_cnv))
print("End date >", time.strftime("%Y/%m/%d %H:%M:%S", end_cnv))

# 学習結果(重みパラメータ)の保存
torch.save(model.state_dict(), "tiny-yolo_2epoch.model")

# lossグラフの作成
plot_graph(losses, EPOCHS)
