# MCMC
MCMC部分算法代码实现。

本代码库实现并集成了若干个比较经典的MCMC算法，主要包括：
- M-H MCMC
- Langevin MCMC
- Ensemble MCMC (Multitry-MCMC)
- Hamilton MCMC (HMC / Hybrid MCMC)
- Parallel Temperature MCMC

本框架共由三部分组成：待采样目标函数、转移核和链。通过在配置文件修改这些组件，你可以实现上述MCMC算法，甚至可以组装自己所喜欢的MCMC算法。

## 1. 待采样目标函数
MCMC算法所采样的目标函数$f_u$。其接受一个$n$维向量，输出对应的密度值。$f_u$不需要经过归一化。

本代码中采用高斯混合模型作为$f_u$以测试MCMC性能。你也可以选择自己编写目标函数。

## 2. 转移核
实现了两种转移核。

### M-H Kernal
传统的M-H转移核。M-H转移核需要一个提议转移核，本代码库中提供的提议转移核有正态随机游走转移核和Langevin转移核，可在配置文件中具体设置。

### HMC Kernal
Hamilton MCMC转移核。其不需要提议转移核。

## 3. 链
实现了两种链。

### Vanilla Chain
记录一条链不同时刻下的状态$\{ X_t \}$。

### P-T Chain
在不同的温度下，同时运行多条链，并进行链与链之间的交换。即Parallel-Temperature算法。此链可以设置多个转移核(和温度设置数一致)。

## 通过修改组件实现不同算法
在配置文件中，可以修改框架的组件以实现不同算法。例如，设置M-H转移核的提议转移核为Langevin转移核即为Langevin MCMC。甚至你可以
配置你所喜欢的MCMC算法，例如可以在P-T Chain下，选择HMC转移核，从而实现HMC算法和Parallel-Temperature算法的结合。