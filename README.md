# TIanchi-CCF-contests-data-intergration
最近参加的算法比赛的整理总结
CCF大赛，的数据挖掘赛道：离散制造工件的质量符合率预测
在给定了P类特征和A类特征进行质量分类预测。
本人在参赛后先进行EDA，数据分析，选取了P5-P10作为建模特征，并进行对数转换，在对P6,7,8三个特征进行计数，构造新特征。
最终模型使用catboost，参数进行过gridsearch后选区最好的，尝试过模型融合，使用xgboost,lightgbm等等，但效果不好，还是用的cbt单模。
线上结果，初赛33名，复赛31名，Top1.5%的成绩。
