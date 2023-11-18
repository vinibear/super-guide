from utils import MyIterator
import torch
from different_models.model1.configs import test_information_path, batch_size
from different_models.model1.configs import pre_trained_model


def main(abs):
    net = torch.load(pre_trained_model)
    # loss = torch.nn.CrossEntropyLoss()
    # test_iter = MyIterator(test_information_path, batch_size)
    net.eval()
    with torch.no_grad():
        """
        以下是使用举例
    
        将需要预测的abstract送入net() 
        然后就得到一个19维的向量
        每一维表示abstract属于该维度所代表的期刊的概率
        具体每个维度代表哪个期刊请见 different_models\model1\configs.py
        以下选择TECS的270号文章送入net()
        TESC在 字典里标号为 3
        即如果输出向量的第四维最大则预测正确
        """
        # label=3 abstract=abstract/TECS/270.txt
        abstract = [abs]
        y = net(abstract,True).to('cpu')
        y = y.numpy().tolist()

        #print(y)
        return y[0]
        """
        输出
        tensor([[3.1832e-04, 1.4807e-04, 4.4304e-04, 9.9280e-01, 1.6743e-03, 4.4610e-04,
                 2.9247e-04, 2.4575e-04, 2.6525e-04, 1.9959e-05, 2.6810e-04, 5.1110e-04,
                 7.7937e-04, 2.3875e-04, 2.9756e-04, 2.7568e-04, 3.0016e-04, 4.7998e-04,
                 1.9284e-04]], device='cuda:0')
        
                 可见标号为3的第四维为 9.9280e-01
                 即输入的abstract属于TESC的概率为99.28%
                 预测正确
        
        """


if __name__ == '__main__':
    a = 'Asymmetric multiprocessor systems are considered power-efficient multiprocessor architectures. Furthermore, efficient task allocation (partitioning) can achieve more energy efficiency at these asymmetric multiprocessor platforms. This article addresses the problem of energy-aware static partitioning of periodic real-time tasks on asymmetric multiprocessor (multicore) embedded systems. The article formulates the problem according to the Dynamic Voltage and Frequency Scaling (DVFS) model supported by the platform and shows that it is an NP-hard problem. Then, the article outlines optimal reference partitioning techniques for each case of DVFS model with suitable assumptions. Finally, the article proposes modifications to the traditional bin-packing techniques and designs novel techniques taking into account the DVFS model supported by the platform. All algorithms and techniques are simulated and compared. The simulation shows promising results, where the proposed techniques reduced the energy consumption by 75\% compared to traditional methods when DVFS is not supported and by 50\% when per-core DVFS is supported by the platform.'
    b='Intellectual Property (IP) reuse is a well known practice in chip design processes. Nowadays, network-on-chips (NoCs) are increasingly used as IP and sold by various vendors to be integrated in a multiprocessor system-on-chip (MPSoC). However, IP reuse exposes the design to IP theft, and an attacker can launch IP stealing attacks against NoC IPs. With the growing adoption of MPSoC, such attacks can result in huge financial losses. In this article, we propose four NoC IP protection techniques using fingerprint embedding: ON-OFF router-based fingerprinting (ORF), ON-OFF link-based fingerprinting (OLF), Router delay-based fingerprinting (RTDF), and Row delay-based fingerprinting (RWDF). ORF and OLF techniques use patterns of ON-OFF routers and links, respectively, while RTDF and RWDF techniques use router delays to embed fingerprints. We show that all of our proposed techniques require much less hardware overhead compared to an existing NoC IP security solution (square spiral routing) and also provide better security from removal and masking attacks. In particular, our proposed techniques require between 40.75% and 48.43% less router area compared to the existing solution. We also show that our solutions do not affect the normal packet latency and hence do not degrade the NoC performance.'
    r=main(b)
    print(r)
    #print(type(r[0]))
