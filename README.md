## コーディング手順
1. nn.conv2d()とかでモデルを定義する	(model.py)
2. 学習データ(COCO)を読み込む					(train.py)
3. データローダを作成する							(train.py)
4. 損失関数を定義する									(function.py)
5. 学習ループを書く										(train.py)
ここまで, training
ここから, test
6. 推論プログラム(.val()して, model(x) するだけ)を書く
7. NMSとか, 後処理を書く							(test.py, detect.py)
8. OpenCVとかで出力画像を表示する			(test.py)

## 流れ
* まずは, 論文とかにあるモデル通りに作って, 学習してみる.
* できれば, mAP算出プログラム入れたい
* そのあと, fine-tuningとかしてみる. 一から学習したものとの比較とかも入れたい.
* そのあと, クラスを減らしてみる
* そのあと, チャネル数を減らしたり, 層を減らしたりしてみる

## コマンドメモ
* クラス数１で転移学習 (yolov3-tiny.weights を学習済みパラメータとして使用)
  python train.py --weights weights/yolov3-tiny.weights \
                  --data_root /home/matsuda/datasets/COCO_car/2014/ \
                  --num_classes 1 \
                  --class_names namefiles/coco_car.names \
                  --trans

* Google_cars (4クラス) の モデルの精度検証
  python test.py --weights weights/yolov3-tiny_car4.pt \
                 --data_root /home/matsuda/datasets/Google_cars \
                 --num_classes 4 \
                 --class_names namefiles/google_cars.names \

* 数エポック学習し終わり, さらにもう一度数エポック学習する場合 (続きから学習)
  -> --finetune
     --weights {いったん学習し終えたモデル}
     --restart
     を入れる

* 一番惜しい検出記録
  第1位
  python detect.py --weights results/20210930_115457/yolo-tiny.pt \
                   --class_names vehicle.names \
                   --num_classes 1 \
                   --image images/COCO_sample_car2.jpg \
                   --nms_thres 0.3 \
                   --conf_thres 0.2
  第2位
  python detect.py --weights results/20210929_152822/yolo-tiny.pt \
                   --class_names vehicle.names \
                   --num_classes 5 \
                   --image images/vehicle_sample3.jpg \
                   --nms_thres 0.4 \
                   --conf_thres 0.4
  第3位
  python detect.py --weights results/20210924_174004/yolo-tiny.pt \
                   --class_names coco_car.names \
                   --num_classes 1 \
                   --image images/COCO_sample_car2.jpg \
                   --nms_thres 0.2 \
                   --conf_thres 0.1


## 備考
* 損失関数は, eriklindernoren の loss.py を使用している.
	今後, 自分で実装したものに差し替える予定.

* データセット・データローダの作成プログラムも
	eriklindernoren のものを使用している.
	今後, 自分で実装したものに差し替える予定.

* weights/yolov3-tiny-old.pt は
  pjreddie の yolov3-tiny.weights を, ultralystic の yolov3 の
  convert (coremltools) を使って変換したもの

* weights/yolov3-tiny.pt は
  ultralytics の releaseからダウンロードしてきたもの

* weightsのタイムスタンプが 9/10 より前のものは, convのbiasがないものなので,
  現在のモデルには使えない.

* 転移学習において, 重みのコピーがうまくできておらず(適切でないところにコピーさ
  れていた), 出力が1層目からっ全然違っていたことが判明.
  現時点で, 1層目の出力は, とりあえず正しくなった. (9/18)
  なお, 正しいかどうかは, erik の yolov3 の推論結果やパラメータと比較して
  確認している.
  yolo層の手前までの出力は正しくなった.

* 9/29 18:00 以前の, COCO_car を使った学習は, ラベルが間違っているので,
  信用できない.

* 9/30 時点で, results/20210930_115457/ の重みを使用して,
  1クラスで conf_thres=0.1, nms_thres=0.3 にしたときが, 複数枚の画像でdetect
  してみた感じで一番いい.

* 11/09 以前のモデルは, Sequentialで書き換える前のモデルなので, パラメータ
  (モデル)を使うことはできない.


## ハイパーパラメータ
* (9/6) 学習率の学習時調整を導入. 調整方法は, ひとまずerikの実装と同じ
  ようにしてみる.

* (10/4) ヒト検出に関しては, conf_thres=0.1 がよさそう

## モデルリスト
* ヒト・信号検出 (最新) : results/20211117_171723/ or results/20211117_180458/
                    (Flip の augmentation あり と なし)
                                  conf_thres=0.1 でいい感じ

  results/20211118_165358/ のやつは, conf_thres=0.2 だといい感じ

  どちらも, iou_thres=0.3 だと二重検出しにくい

* 車両検出 (最新) : results/20211123_172801/
  -> weights/yolov3-tiny_car4.pt

* 車両検出 (1クラス; 最新) : results/20211201_125135/

* 車両検出 (1クラス separable; 最新) : results/20220120_161120/
  (1/20. これ以前の sep は, DWとPWしたあとにBN, LRelu をしている.
   1/20 以降の sep は, DW, BN, LRelu, PW, BN, LRelu になっている

* 車両検出 (さらに最新) : results/20220124_014624/
  最初の層も separable にしたやつ. 意外と精度落ちなかった.


## Gitでの衝突を避けるための, 編集ルール
* train.py は, dlbox のみで編集する.
* model.py は, dlbox のみで編集する.
* plot や, パラメータ閲覧などのデバグ系, ツール系は, ホストマシン(cardinal) で
  編集する.
