YOLO層が受け取る入力は, クラス数が80とすると
batch_size * ((80+5) * 3)
のTensor

1. これをまず, 80+5 が 3セット（がbatch_size個) あるというデータ構造にする.

2. 1セットの80+5のデータにつき, 1つのanchorボックスが対応する.

3. 特徴マップの値と, anchorボックスの値から, 矩形の大きさを計算する
	 入力のx[85] は x[0] x[1] x[2] x[3] x[4]	 x[5] x[6] ... x[84]
									 tx		ty	 tw		th	 tconf	t0	 t1				t79
	 というふうな配列になっている

	 入力の80+5が, tx, ty, tw, th, tconf, t0, t1, ..., t79 とし,
	 anchorボックスの大きさを aw, ah とすると, 矩形の 幅bw, 高さbh は,

	 bw = exp(tw) * aw
	 bh = exp(th) * ah

	 また, 13x13 のグリッドでの座標 gx, gy (0 <= gx, gy < 13) を使って,
	 anchorボックスの中心座標 (bx, by) は

	 bx = gx + sigmoid(tx)
	 by = gy + sigmoid(ty)

	 で算出する. gx, gy は 0~13 まで, 169通りの矩形（の中心）を算出するらしい.

4. 信頼度 p は
	 p = sigmoid(tconf)
	 で算出する

5. それぞれのクラス確率 p0, p1, ..., p79 は
	 p0 = sigmoid(t0)
	 p1 = sigmoid(t1)
				...
	 p79 = sigmoid(t79)
	 で算出する.

	 この bw, bh, bx, by, p, p0~p79 が, 3セット分算出される

	 1つのグリッドにつき3つのボックスを予測. それが13x13グリッドあるので
	 3x13x13 個のボックスを, 1つのYOLO層で予測することになる

<損失の計算>
6. クラス確率のloss (t0~t79) は, Binery Cross Entropy で算出
