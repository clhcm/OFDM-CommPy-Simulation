# OFDM-CommPy-Simulation
This repository contains Python code for simulating an OFDM system using CommPy. Includes modulation (BPSK/QPSK), channel modeling (AWGN), and bit error rate (BER) evaluation.）

一、利用commpy库的OFDM调制，主要实现功能：

1.调制与解调，实现的调制方式可有BPSK、QPSK、8PSK、16QAM、64QAM，利用离散傅里叶反变换（IDFT）和变换(DFT)实现多个载波的调制与解调。

2.信道模拟，支持模拟加性高斯白噪声（AWGN）信道和多径衰落信道，用于测试信号在噪声环境下的性能。

3.循环前缀（CP）处理，用于增强 OFDM 系统对多径衰落的抗性，预留了在发送端插入和接收端去除循环前缀的功能。

4.性能分析与信号处理，使用信噪比（SNR）参数 SNRdb 来模拟不同信道条件下系统的传输性能，可通过性能分析，如误码率 (BER) 评估 OFDM 系统在不同信道下的表现，以及星座图。

二、根据OFDM系统原理设计仿真实现具体过程如下：

1.产生比特流

2.比特流映射到选定调制方式的复数符号

3.插入导频和数据子载波，实现串并转换

4.IFFT，并行的子载波符号转换为时域信号

5.添加循环前缀，对抗ISI

6.经过信道

7.去除循环前缀后再FFT

8.均衡

9.反映射得到比特流
