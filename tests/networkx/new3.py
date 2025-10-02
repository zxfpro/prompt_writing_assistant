```python

__doc__ = """

READY:

datasets ML models monitor utils(evals llm_server process)

TODO:

custom_models fine_turning plightning pth2atc pth2onnx pth2tensorrt signature train train_tools(managers models transformers)

NOTES:

pytorch lightning ; ignite

训练 一定要固定随机种子

Dask pandas高级版  

dython 用来解决数值型数据与类别型数据之间的相关矩阵问题库   

cuML

Pydantic # FastAPI 开发中数据校验利器

featuretools 自动化特征工程

optuna 自动化调参工具库

deltarpm(增量下载包)  

gin 机器学习实验设置大量参数的库 pip install gin-config

numba  

conn

  
  
  

"""

```


```python

"""
N-BEATS Model.
"""
from typing import Tuple
import numpy as np
import torch as t


class NBeatsBlock(t.nn.Module):
    """
    N-BEATS block which takes a basis function as an argument.
    """

    def __init__(self,
                 input_size,
                 theta_size: int,
                 basis_function: t.nn.Module,
                 layers: int,
                 layer_size: int):
        """
        N-BEATS block.

        :param input_size: Insample size.
        :param theta_size:  Number of parameters for the basis function.
        :param basis_function: Basis function which takes the parameters and produces backcast and forecast.
        :param layers: Number of layers.
        :param layer_size: Layer size.
        """
        super().__init__()
        self.layers = t.nn.ModuleList([t.nn.Linear(in_features=input_size, out_features=layer_size)] +
                                      [t.nn.Linear(in_features=layer_size, out_features=layer_size)
                                       for _ in range(layers - 1)])
        self.basis_parameters = t.nn.Linear(in_features=layer_size, out_features=theta_size)
        self.basis_function = basis_function

    def forward(self, x: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        block_input = x
        for layer in self.layers:
            block_input = t.relu(layer(block_input))
        basis_parameters = self.basis_parameters(block_input)
        return self.basis_function(basis_parameters)


class NBeats(t.nn.Module):
    """
    N-Beats Model.
    """

    def __init__(self, blocks: t.nn.ModuleList):
        super().__init__()
        self.blocks = blocks

    def forward(self, x: t.Tensor, input_mask: t.Tensor) -> t.Tensor:
        residuals = x.flip(dims=(1,))
        input_mask = input_mask.flip(dims=(1,))
        forecast = x[:, -1:]
        for i, block in enumerate(self.blocks):
            backcast, block_forecast = block(residuals)
            residuals = (residuals - backcast) * input_mask
            forecast = forecast + block_forecast

        return forecast


class GenericBasis(t.nn.Module):
    """
    Generic basis function.
    """

    def __init__(self, backcast_size: int, forecast_size: int):
        super().__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size

    def forward(self, theta: t.Tensor):
        return theta[:, :self.backcast_size], theta[:, -self.forecast_size:]


class TrendBasis(t.nn.Module):
    """
    Polynomial function to model trend.
    """

    def __init__(self, degree_of_polynomial: int, backcast_size: int, forecast_size: int):
        super().__init__()
        self.polynomial_size = degree_of_polynomial + 1  # degree of polynomial with constant term
        self.backcast_time = t.nn.Parameter(
            t.tensor(np.concatenate([np.power(np.arange(backcast_size, dtype=np.float) / backcast_size, i)[None, :]
                                     for i in range(self.polynomial_size)]), dtype=t.float32),
            requires_grad=False)
        self.forecast_time = t.nn.Parameter(
            t.tensor(np.concatenate([np.power(np.arange(forecast_size, dtype=np.float) / forecast_size, i)[None, :]
                                     for i in range(self.polynomial_size)]), dtype=t.float32), requires_grad=False)

    def forward(self, theta: t.Tensor):
        backcast = t.einsum('bp,pt->bt', theta[:, self.polynomial_size:], self.backcast_time)
        forecast = t.einsum('bp,pt->bt', theta[:, :self.polynomial_size], self.forecast_time)
        return backcast, forecast


class SeasonalityBasis(t.nn.Module):
    """
    Harmonic functions to model seasonality.
    """

    def __init__(self, harmonics: int, backcast_size: int, forecast_size: int):
        super().__init__()
        self.frequency = np.append(np.zeros(1, dtype=np.float32),
                                   np.arange(harmonics, harmonics / 2 * forecast_size,
                                             dtype=np.float32) / harmonics)[None, :]
        backcast_grid = -2 * np.pi * (
                np.arange(backcast_size, dtype=np.float32)[:, None] / forecast_size) * self.frequency
        forecast_grid = 2 * np.pi * (
                np.arange(forecast_size, dtype=np.float32)[:, None] / forecast_size) * self.frequency
        self.backcast_cos_template = t.nn.Parameter(t.tensor(np.transpose(np.cos(backcast_grid)), dtype=t.float32),
                                                    requires_grad=False)
        self.backcast_sin_template = t.nn.Parameter(t.tensor(np.transpose(np.sin(backcast_grid)), dtype=t.float32),
                                                    requires_grad=False)
        self.forecast_cos_template = t.nn.Parameter(t.tensor(np.transpose(np.cos(forecast_grid)), dtype=t.float32),
                                                    requires_grad=False)
        self.forecast_sin_template = t.nn.Parameter(t.tensor(np.transpose(np.sin(forecast_grid)), dtype=t.float32),
                                                    requires_grad=False)

    def forward(self, theta: t.Tensor):
        params_per_harmonic = theta.shape[1] // 4
        backcast_harmonics_cos = t.einsum('bp,pt->bt', theta[:, 2 * params_per_harmonic:3 * params_per_harmonic],
                                          self.backcast_cos_template)
        backcast_harmonics_sin = t.einsum('bp,pt->bt', theta[:, 3 * params_per_harmonic:], self.backcast_sin_template)
        backcast = backcast_harmonics_sin + backcast_harmonics_cos
        forecast_harmonics_cos = t.einsum('bp,pt->bt',
                                          theta[:, :params_per_harmonic], self.forecast_cos_template)
        forecast_harmonics_sin = t.einsum('bp,pt->bt', theta[:, params_per_harmonic:2 * params_per_harmonic],
                                          self.forecast_sin_template)
        forecast = forecast_harmonics_sin + forecast_harmonics_cos

        return backcast, forecast


"""
Shortcut functions to create N-BEATS models.
"""


def interpretable(input_size: int,
                  output_size: int,
                  trend_blocks: int,
                  trend_layers: int,
                  trend_layer_size: int,
                  degree_of_polynomial: int,
                  seasonality_blocks: int,
                  seasonality_layers: int,
                  seasonality_layer_size: int,
                  num_of_harmonics: int):
    """
    Create N-BEATS interpretable model.
    """
    trend_block = NBeatsBlock(input_size=input_size,
                              theta_size=2 * (degree_of_polynomial + 1),
                              basis_function=TrendBasis(degree_of_polynomial=degree_of_polynomial,
                                                        backcast_size=input_size,
                                                        forecast_size=output_size),
                              layers=trend_layers,
                              layer_size=trend_layer_size)
    seasonality_block = NBeatsBlock(input_size=input_size,
                                    theta_size=4 * int(
                                        np.ceil(num_of_harmonics / 2 * output_size) - (num_of_harmonics - 1)),
                                    basis_function=SeasonalityBasis(harmonics=num_of_harmonics,
                                                                    backcast_size=input_size,
                                                                    forecast_size=output_size),
                                    layers=seasonality_layers,
                                    layer_size=seasonality_layer_size)

    return NBeats(t.nn.ModuleList(
        [trend_block for _ in range(trend_blocks)] + [seasonality_block for _ in range(seasonality_blocks)]))


def generic(input_size: int, output_size: int,
            stacks: int, layers: int, layer_size: int):
    """
    Create N-BEATS generic model.
    """
    return NBeats(t.nn.ModuleList([NBeatsBlock(input_size=input_size,
                                               theta_size=input_size + output_size,
                                               basis_function=GenericBasis(backcast_size=input_size,
                                                                           forecast_size=output_size),
                                               layers=layers,
                                               layer_size=layer_size)
                                   for _ in range(stacks)]))


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = torch.nn.Linear(10, 10)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(10, 5)

    def forward(self, x):
        x = self.relu(self.net1(x))
        return self.net2(x)


class CGAN(Model):
    condition = True
    name = 'CGAN'

    def __init__(self, network, device='cpu'):
        super(CGAN, self).__init__(network, device=device)
        self.discriminator = network.discriminator
        self.generator = network.generator
        self.dis_loss_function = nn.BCELoss()
        self.device = device

    def generate(self, batch_size):
        noise = torch.randn(batch_size, 1)
        out = self.generator.forward(noise)
        return out

    def train(self, real_batch, condition):
        batch_size = real_batch.size()[0]
        true_target = torch.FloatTensor([1.0] * batch_size, device=self.device).view(-1, 1)
        fake_target = torch.FloatTensor([0.0] * batch_size, device=self.device).view(-1, 1)

        # 训练判别器

        # print(true_target.size(), "true_target")
        true_dis_out = self.discriminator.forward(real_batch, condition)
        # print(true_dis_out.size(), "true_dis_out")
        true_dis_loss = self.dis_loss_function(true_dis_out, true_target)

        noise_inputs = torch.randn(batch_size, 1)
        generator_out = self.generator.forward(noise_inputs, condition)
        # print(generator_out.size(), "generator_out")
        fake_dis_out = self.discriminator.forward(generator_out, condition)
        fake_dis_loss = self.dis_loss_function(fake_dis_out, fake_target)

        dis_loss = true_dis_loss + fake_dis_loss
        network.dis_optimiser.zero_grad()
        dis_loss.backward()
        network.dis_optimiser.step()

        # 训练生成器
        noise_inputs = torch.randn(batch_size, 1)
        generator_out = self.generator.forward(noise_inputs, condition)
        fake_dis_out = self.discriminator.forward(generator_out, condition)
        fake_dis_loss = self.dis_loss_function(fake_dis_out, fake_target)

        network.ge_optimiser.zero_grad()
        fake_dis_loss.backward()
        network.ge_optimiser.step()

        return true_dis_loss, fake_dis_loss




```



```python
from llama_index.core import PromptTemplate
from quickuse.llms import get_llm
import pandas as pd
import io

template = """

你是一个pandas的大师, 你需要根据你看到的df.sample(4) 的数据样例, 和用户的诉求, 编写对应的操作python代码来实现功能.

假设你可以直接操作名为df 的变量,它就是你看到的DataFrame.

你可以将你的输出打包成函数,这样用户更方便使用

  

数据样例(df.sample(4)):

---

{data_sample}

---

数据格式(df.info()):

---

{data_info}

---

用户诉求:

{prompt}

---

输出python程序:

"""

  
  

def get_df_info(df):
	
	# 得到df_info 的信息
	
	buffer = io.StringIO()
	
	df.info(buf=buffer)
	
	result = buffer.getvalue()
	
	buffer.close()
	
	return result

  

def pandasmaster(multi_line_input):
	
	df_path,prompt = multi_line_input.split('&')
	
	df = pd.read_csv(df_path)
	
	llm = get_llm()
	
	qa_template = PromptTemplate(template=template)
	
	sample_df = df.sample(4).to_markdown()
	
	df_info = get_df_info(df)
	
	response = llm.complete(qa_template.format(data_sample=sample_df, prompt=prompt, data_info=df_info))
	
	return response.text

  
  
  

if __name__ == "__main__":

	import sys
	
	multi_line_input = sys.stdin.read()
	
	print(pandasmaster(multi_line_input))

```