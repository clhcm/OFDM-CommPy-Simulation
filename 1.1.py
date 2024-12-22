import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import commpy as cpy

K = 64 # OFDM子载波数量
CP = K//4 #25%的循环前缀长度
P = 8  # 导频数
pilotValue = 3+3j  # 导频格式
Modulation_type = ('QAM16') #调制方式，可选BPSK、QPSK、8PSK、QAM16、QAM64
channel_type ='awgn' # 信道类型，可选awgn
SNRdb = 25  # 接收端的信噪比（dB）
allCarriers = np.arange(K)  # 子载波编号 ([0, 1, ... K-1]),生成一个从0到K-1的数组。numpy库中的arange()函数创建一个均匀分布的一维数组
pilotCarrier = allCarriers[::K//P]  # 每间隔P个子载波一个导频,Python中的切片语法，用于从数组中选择特定的元素,::K//P表示从allCarriers数组中每隔 K//P 个元素选择一个即[0, 8, 16, 24, 32, 40, 48, 56]。
# 为了方便信道估计，将最后一个子载波也作为导频
pilotCarriers = np.hstack([pilotCarrier, np.array([allCarriers[-1]])])#pilotCarriers = [0, 8, 16, 24, 32, 40, 48, 56, 63]
P = P+1 # 导频的数量也需要加1

# 可视化数据和导频的插入方式
dataCarriers = np.delete(allCarriers, pilotCarriers)#从 allCarriers 数组中删除 pilotCarriers 中的元素，赋给dataCarriers
plt.figure(figsize=(12, 3))
plt.plot(pilotCarriers, np.zeros_like(pilotCarriers), 'bo', label='pilot')
plt.plot(dataCarriers, np.zeros_like(dataCarriers), 'ro', label='data')
plt.legend(fontsize=10, ncol=2)#fontsize=10 设置图例字体的大小为10,ncol=2 表示图例中的条目分为2列显示。
plt.xlim((0, K))
plt.ylim((-0.3, 0.3))
plt.xlabel('Carrier')
plt.yticks([])#设置Y轴的刻度为空
plt.grid(True)#启用网格线
#plt.savefig('carrier.png')#将当前绘制的图保存为名为 'carrier.png' 的图片文件

m_map = {"BPSK": 1, "QPSK": 2, "8PSK": 3, "QAM16": 4, "QAM64": 6}
mu = m_map[Modulation_type]
payloadBits_per_OFDM = len(dataCarriers)*mu  # 每个 OFDM 符号能够承载的比特数，即有效载荷的位数。
# 定制调制方式
def Modulation(bits):#def是Python中定义函数的关键字,Modulation 是函数的名称,(bits) 是函数的参数,这通常是一个二进制位的序列
    if Modulation_type == "QPSK":
        PSK4 = cpy.PSKModem(4) #创建一个 PSKModem实例，处理QPSK调制,4相位键控
        symbol = PSK4.modulate(bits)#调用 modulate(bits) 方法，将比特流转换为相应的符号
        return symbol
    elif Modulation_type == "QAM64":
        QAM64 = cpy.QAMModem(64)
        symbol = QAM64.modulate(bits)
        return symbol
    elif Modulation_type == "QAM16":
        QAM16 = cpy.QAMModem(16)
        symbol = QAM16.modulate(bits)
        return symbol
    elif Modulation_type == "8PSK":
        PSK8 = cpy.PSKModem(8)
        symbol = PSK8.modulate(bits)
        return symbol
    elif Modulation_type == "BPSK":
        BPSK = cpy.PSKModem(2)
        symbol = BPSK.modulate(bits)
        return symbol
# 定义解调方式
def DeModulation(symbol):
    if Modulation_type == "QPSK":
        PSK4 = cpy.PSKModem(4)
        bits = PSK4.demodulate(symbol, demod_type='hard')#demod_type='hard' 表示使用硬判决解调（hard decision demodulation），这是一种根据接收到的符号值直接确定比特值的方法。
        return bits
    elif Modulation_type == "QAM64":
        QAM64 = cpy.QAMModem(64)
        bits = QAM64.demodulate(symbol, demod_type='hard')
        return bits
    elif Modulation_type == "QAM16":
        QAM16 = cpy.QAMModem(16)
        bits = QAM16.demodulate(symbol, demod_type='hard')
        return bits
    elif Modulation_type == "8PSK":
        PSK8 = cpy.PSKModem(8)
        bits = PSK8.demodulate(symbol, demod_type='hard')
        return bits
    elif Modulation_type == "BPSK":
        BPSK = cpy.PSKModem(2)
        bits = BPSK.demodulate(symbol, demod_type='hard')
        return bits

# 可视化信道冲击响应，仿真信道
# the impulse response of the wireless channel
channelResponse = np.array([1, 0, 0.3+0.3j])#表示信道冲激响应
H_exact = np.fft.fft(channelResponse, K)#频域响应，np.fft.fft() 是 NumPy 提供的 FFT 函数，用于计算输入信号的 离散傅里叶变换（DFT）
#plt.figure(figsize=(10, 6))
#plt.plot(allCarriers, abs(H_exact),'b-')#*args提示是在告诉你，这些函数可以接受额外的参数
#plt.xlabel('Subcarrier index')
#plt.ylabel('$|H(f)|$')
#plt.grid(True)
#plt.xlim((0, K-1))
# 定义信道
def add_awgn(x_s, snrDB):#根据输入信号和信噪比来算噪声
    data_pwr = np.mean(abs(x_s**2))
    noise_pwr = data_pwr/(10**(snrDB/10))
    noise = 1/np.sqrt(2) * (np.random.randn(len(x_s)) + 1j *np.random.randn(len(x_s))) * np.sqrt(noise_pwr)
 #从 np.random.randn() 生成的正态分布随机数的方差是 1 ,为了符合复高斯噪声的定义（实部和虚部的方差都是 0.5），我们乘以1/根号2，又为了将噪声功率提到所定义的，就又×定义的噪声的根号
    return x_s + noise, noise_pwr
def channel(in_signal, SNRdb, channel_type="awgn"):
    channelResponse = np.array([1, 0, 0.3+0.3j]) #随意仿真信道冲击响应
    if channel_type == "random":
        convolved = np.convolve(in_signal, channelResponse)
        out_signal, noise_pwr = add_awgn(convolved, SNRdb)#调用 add_awgn 函数，在卷积结果上添加 AWGN 噪声，模拟信道的噪声环境
    elif channel_type == "awgn":
        out_signal, noise_pwr = add_awgn(in_signal, SNRdb)
    return out_signal, noise_pwr

# 5.1 产生比特流
bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM, ))
#生成一个长度为 payloadBits_per_OFDM 的随机二进制比特序列，每个比特的值是 0 或 1，且两者出现的概率相等（50%)
# 5.2 比特信号调制
QAM_s = Modulation(bits)#串行比特流通过 Modulation(bits) 转换为复数符号（QAM_s）
# 5.3 插入导频和数据，生成OFDM符号
def OFDM_symbol(QAM_payload):
    symbol = np.zeros(K, dtype=complex) # 初始化一个长度为k(OFDM子载波数量)的数组，数据类型是复数（dtype=complex）
    symbol[pilotCarriers] = pilotValue  # 在导频位置插入导频
    symbol[dataCarriers] = QAM_payload  # 在数据位置插入数据，将串行符号 QAM_s 分配到不同的子载波上，形成多个并行数据流。
    return symbol
OFDM_data = OFDM_symbol(QAM_s) #生成OFDM符号

# 5.4 快速傅里叶逆变换
def IDFT(OFDM_data):
    return np.fft.ifft(OFDM_data)
OFDM_time = IDFT(OFDM_data)
# 5.5 添加循环前缀
def addCP(OFDM_time):
    cp = OFDM_time[-CP:] #取 OFDM_time的最后CP个采样点作为循环前缀
    return np.hstack([cp, OFDM_time])
OFDM_withCP = addCP(OFDM_time)

# 5.6 经过信道
OFDM_TX = OFDM_withCP
OFDM_RX = channel(OFDM_TX, SNRdb, "awgn")[0]

plt.figure(figsize=(10,6))
plt.plot(abs(OFDM_TX), label='TX signal')
plt.plot(abs(OFDM_RX), label='RX signal')
plt.legend(fontsize=10)
plt.xlabel('Time'); plt.ylabel('$|x(t)|$');
plt.grid(True)
# plt.savefig('tran-receiver.png')

# 5.7 接收端，去除循环前缀
def removeCP(signal):
    return signal[CP:(CP + K)]
OFDM_RX_noCP = removeCP(OFDM_RX)


# 5.8 快速傅里叶变换
def DFT(OFDM_RX):
    return np.fft.fft(OFDM_RX)

OFDM_demod = DFT(OFDM_RX_noCP)

# 5.9 信道估计
def channelEstimate(OFDM_demod):
    pilots = OFDM_demod[pilotCarriers]  # 取导频处的数据
    Hest_at_pilots = pilots/pilotValue  # LS信道估计
    # 在导频载波之间进行插值以获得估计，然后利用插值估计得到数据下标处的信道响应
    Hest_abs = interpolate.interp1d(pilotCarriers, abs(Hest_at_pilots), kind='linear')(allCarriers)
    Hest_phase = interpolate.interp1d(pilotCarriers, np.angle(Hest_at_pilots), kind='linear')(allCarriers)
#调用scipy.interpolate.interp1d函数 创建一个插值函数，(allCarriers)表示调用上面生成的插值函数，并传入新的输入值allCarriers
    Hest = Hest_abs * np.exp(1j * Hest_phase)
#使用 interpolate.interp1d 对幅度和相位分别做线性插值
    plt.figure(figsize=(10, 6))
    plt.plot(allCarriers, abs(H_exact), label='Correct Channel')
    plt.scatter(pilotCarriers, abs(Hest_at_pilots), label='Pilot estimates')
    plt.plot(allCarriers, abs(Hest), label='Estimated channel via interpolation')
    plt.grid(True);
    plt.xlabel('Carrier index');
    plt.ylabel('$|H(f)|$');
    plt.legend(fontsize=10)
    plt.ylim(0, 2)
    #plt.savefig('信道响应估计.png')
    return Hest

Hest = channelEstimate(OFDM_demod)

# 5.10 均衡
def equalize(OFDM_demod, Hest):
    return OFDM_demod / Hest
equalized_Hest = equalize(OFDM_demod, Hest)

# 5.10 从均衡后的信号中获取数据位置的数据
def get_payload(equalized):
    return equalized[dataCarriers]
QAM_est = get_payload(equalized_Hest)
# 可视化均衡后的星座图
plt.figure(figsize=(10, 6))
plt.plot(QAM_est.real, QAM_est.imag, 'bo')
plt.plot(QAM_s.real, QAM_s.imag, 'ro')

plt.grid(True)
plt.xlabel('Real part')
plt.ylabel('Imaginary Part')
plt.title("Received constellation")
#plt.savefig('map.png')

# 5.11 反映射，解调
bits_est = DeModulation(QAM_est)
# 5.12 计算误比特率
print ("误比特率BER： ", np.sum(abs(bits-bits_est))/len(bits))

plt.show()