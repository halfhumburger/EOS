# EOS
化工热力学第一次大作业
基于机器学习视角修正EOS
请大家及时上传项目文件，我们可以在这里实时跟进进度！

思路一：

数据来源（data resources）：
实际气体数据和高精度方程导出的PVT数据
水和空气在对应T下的物性参数以及含氢键有机物的基本物性数据（The Yaws handbook of physical properties for hydrocarbons -- Yaws, Carl L）

优化对象：SRK方程（暂定）

原理：对于cubic EOS来说，每一个方程会对应一组a、b=f(Tc,Pc)，a,b表达式的系数是固定的（对于一个特定的方程）
对于不同的实际气体来说，系数不一定是固定的

优化方案：利用高精度的PVT计算出a或者b，拟合a或者b对于“分子”的曲线
