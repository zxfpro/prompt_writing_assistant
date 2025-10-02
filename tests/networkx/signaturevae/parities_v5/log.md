# version 4

1 本方法使用原始tensorflow 进行生成 使用了signature化 和遗传算法进行反signature 使用多进程提升生成效率

2 提供了两个方法 train ,generater 使用方式见demo. datadealer 存放处理数据的方法,tf_model 存放模型的方法

3 本方法参数固定,不需要调参,学习率固定在tf_model中,其他参数也对应固定在相应模块内

4 如果调整参数,需要注意 train中的M 和W 应该对应 n_point 21 和 5 level 是signature的等级