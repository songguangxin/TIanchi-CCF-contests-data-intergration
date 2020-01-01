# TIanchi-CCF-contests-data-intergration
## CCF大赛Top31经验分享
[CCF大赛:离散制造中的工件质量符合率预测](https://www.datafountain.cn/competitions/351)            
运行环境：win10 <br> 
         python3.7   
         anaconda spyder                 
在给定了P类特征和A类特征进行质量分类预测。
本人在参赛后先进行EDA，数据分析，选取了P5-P10作为建模特征，并进行对数转换，在对P6,7,8三个特征进行计数，构造新特征，对计数特征进行标准化和对数变换。
最终模型使用catboost，参数进行过gridsearch后选取最好的参数，5折交叉验证，尝试过模型融合，使用xgboost,lightgbm等等，但效果不好，还是用的cbt单模。
线上结果，初赛33名，复赛31名，Top1.5%的成绩。
