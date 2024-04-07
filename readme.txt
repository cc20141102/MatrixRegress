1. 先配置config.son文件
    data_path: 总的训练集路径,
    input_shape: [7, 100, 100],
    output_shape: [3, 1000, 1000],
    pred_path: 仅预测的数据集目录（无标签）,
    save_path: 预测结果保存目录,
    test_path: 用于评估的数据集目录（有标签）,
    result_csv: 评估结果的csv文件路径

2. evaluate.py 用于预测或者评估
    python evaluate.py -mode=pred  # 预测
    python evaluate.py -mode=test  # 评估，结果保存在result_csv中

3. datasplit.py 用于训练前划分数据集
    python datasplit.py  # 在与000同级目录下产生000_split目录

4. train.py 用于训练数据
    # 从0训练
    python train.py -batch_size=28 -warm_start=0 -epochs=100
    
    # 接着某次训练后训练
    python train.py -batch_size=28 -warm_start=1 -checkpoint=./weights/AED.pth -epochs=100

5. 模型架构
    输入：[7, 100, 100]
    encoder 4层反卷积块得到 [3, 1600, 1600]
        a.《转置卷积+跳过连接+卷积》输出 [6, 200, 200]
	b.《转置卷积+跳过连接+卷积》输出 [5, 400, 400]
	c.《转置卷积+跳过连接+卷积》输出 [4, 800, 800]
	d.《转置卷积+跳过连接+卷积》输出 [3, 1600, 1600]
    decoder 2层卷积块得到 [3, 1000, 1000]
	a.《卷积*2+插值》输出 [3, 1000, 1000]
	b. encoder输出插值得 [3, 1000, 1000]
	c. 《卷积*2》对concat(a,b)计算输出 [3, 1000, 1000]

