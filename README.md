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
                  --trans

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


## ハイパーパラメータ
* (9/6) 学習率の学習時調整を導入. 調整方法は, ひとまずerikの実装と同じ
  ようにしてみる.
