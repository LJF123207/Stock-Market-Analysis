# Copyright (c) Microsoft Corporation.  # 版权声明
# Licensed under the MIT License.  # MIT 许可说明

import numpy as np  # 导入 numpy 用于数值运算
import pandas as pd  # 导入 pandas 处理表格数据
import lightgbm as lgb  # 导入 lightgbm 库
from typing import List, Text, Tuple, Union  # 导入类型标注工具
from ...model.base import ModelFT  # 引入基础模型类 ModelFT
from ...data.dataset import DatasetH  # 引入带 handler 的数据集类
from ...data.dataset.handler import DataHandlerLP  # 引入数据处理器基类
from ...model.interpret.base import LightGBMFInt  # 引入 LightGBM 可解释性基类
from ...data.dataset.weight import Reweighter  # 引入样本重加权接口
from qlib.workflow import R  # 引入 Recorder 全局实例


class LGBModel(ModelFT, LightGBMFInt):  # 定义 LightGBM 模型封装类
    """LightGBM Model"""  # 类的文档说明

    def __init__(self, loss="mse", early_stopping_rounds=50, num_boost_round=1000, **kwargs):  # 初始化模型参数
        if loss not in {"mse", "binary"}:  # 检查损失类型是否合法
            raise NotImplementedError  # 不支持其他损失类型
        self.params = {"objective": loss, "verbosity": -1}  # 设置基本 LightGBM 参数
        self.params.update(kwargs)  # 合并用户自定义参数
        self.early_stopping_rounds = early_stopping_rounds  # 提前停止轮数
        self.num_boost_round = num_boost_round  # 训练轮次
        self.model = None  # 保存训练后的模型

    def _prepare_data(self, dataset: DatasetH, reweighter=None) -> List[Tuple[lgb.Dataset, str]]:  # 准备 LightGBM 数据集
        """
        The motivation of current version is to make validation optional
        - train segment is necessary;
        """
        ds_l = []  # 存放 (Dataset, segment_name) 列表
        assert "train" in dataset.segments  # 必须存在训练段
        for key in ["train", "valid"]:  # 依次处理 train 与 valid
            if key in dataset.segments:  # 若配置了该分段
                df = dataset.prepare(key, col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)  # 取特征与标签
                if df.empty:  # 检查数据是否为空
                    raise ValueError("Empty data from dataset, please check your dataset config.")  # 提示配置错误
                x, y = df["feature"], df["label"]  # 拆分特征与标签

                # Lightgbm need 1D array as its label
                if y.values.ndim == 2 and y.values.shape[1] == 1:  # 标签需为一维
                    y = np.squeeze(y.values)  # 压缩为一维数组
                else:
                    raise ValueError("LightGBM doesn't support multi-label training")  # 多标签不支持

                if reweighter is None:  # 若不重加权
                    w = None  # 样本权重为空
                elif isinstance(reweighter, Reweighter):  # 支持 Reweighter 实例
                    w = reweighter.reweight(df)  # 通过 reweighter 获取权重
                else:
                    raise ValueError("Unsupported reweighter type.")  # 其他类型不支持
                ds_l.append((lgb.Dataset(x.values, label=y, weight=w), key))  # 构造 LightGBM 数据对象
        return ds_l  # 返回数据集列表

    def fit(  # 训练模型
        self,
        dataset: DatasetH,
        num_boost_round=None,
        early_stopping_rounds=None,
        verbose_eval=20,
        evals_result=None,
        reweighter=None,
        **kwargs,
    ):
        if evals_result is None:  # 处理默认参数
            evals_result = {}  # in case of unsafety of Python default values  # 避免可变对象默认值陷阱
        ds_l = self._prepare_data(dataset, reweighter)  # 准备训练与验证数据
        ds, names = list(zip(*ds_l))  # 拆分数据集与名称
        early_stopping_callback = lgb.early_stopping(  # 创建提前停止回调
            self.early_stopping_rounds if early_stopping_rounds is None else early_stopping_rounds  # 使用传入或默认值
        )
        # NOTE: if you encounter error here. Please upgrade your lightgbm  # 如有错误建议升级 LightGBM
        verbose_eval_callback = lgb.log_evaluation(period=verbose_eval)  # 设置日志输出频率
        evals_result_callback = lgb.record_evaluation(evals_result)  # 记录评估结果
        self.model = lgb.train(  # 调用 LightGBM 训练接口
            self.params,  # 传入训练参数
            ds[0],  # training dataset  # 第一项为训练集
            num_boost_round=self.num_boost_round if num_boost_round is None else num_boost_round,  # 设置训练轮数
            valid_sets=ds,  # 传入验证集列表
            valid_names=names,  # 验证集名称
            callbacks=[early_stopping_callback, verbose_eval_callback, evals_result_callback],  # 注册回调
            **kwargs,
        )
        for k in names:  # 遍历每个数据分段
            for key, val in evals_result[k].items():  # 遍历评估指标
                name = f"{key}.{k}"  # 构造指标名称
                for epoch, m in enumerate(val):  # 逐轮日志
                    R.log_metrics(**{name.replace("@", "_"): m}, step=epoch)  # 将指标写入 Recorder

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):  # 预测接口
        if self.model is None:  # 确保已经训练
            raise ValueError("model is not fitted yet!")  # 未训练则报错
        x_test = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)  # 准备预测特征
        return pd.Series(self.model.predict(x_test.values), index=x_test.index)  # 返回预测结果并保留索引

    def finetune(self, dataset: DatasetH, num_boost_round=10, verbose_eval=20, reweighter=None):  # 微调接口
        """
        finetune model

        Parameters
        ----------
        dataset : DatasetH
            dataset for finetuning
        num_boost_round : int
            number of round to finetune model
        verbose_eval : int
            verbose level
        """
        # Based on existing model and finetune by train more rounds  # 基于现有模型继续训练
        dtrain, _ = self._prepare_data(dataset, reweighter)  # pylint: disable=W0632  # 仅需要训练集
        if dtrain.empty:  # 检查是否取到数据
            raise ValueError("Empty data from dataset, please check your dataset config.")  # 若为空则报错
        verbose_eval_callback = lgb.log_evaluation(period=verbose_eval)  # 配置日志输出
        self.model = lgb.train(  # 使用 LightGBM 继续训练
            self.params,  # 使用原参数
            dtrain,  # 训练集
            num_boost_round=num_boost_round,  # 训练轮数
            init_model=self.model,  # 以已有模型为初始模型
            valid_sets=[dtrain],  # 仅使用训练集做验证
            valid_names=["train"],  # 验证集名称
            callbacks=[verbose_eval_callback],  # 仅记录日志
        )
