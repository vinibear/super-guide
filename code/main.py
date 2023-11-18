import example
import timeModel


def main(abs, keyw, r):
    a = example.main(abs)
    b = timeModel.main(keyw)
    f = []
    for i in range(19):
        f.append(a[i] * r + b[i] * (1 - r))
    return f


def final(f):
    label_dir = {'JACM': 0, 'TAAS': 1, 'TACO': 2, 'TECS': 3, 'TKDD': 4,
                 'TOCHI': 5, 'TODAES': 6, 'TODS': 7, 'TWEB': 8, 'TOG': 9,
                 'TOIS': 10, 'TOIT': 11, 'TOMM': 12, 'TOPLAS': 13, 'TOPS': 14,
                 'TOS': 15, 'TOSEM': 16, 'TOSN': 17, 'TRETS': 18}
    jours = list(label_dir.keys())
    score_dir={}
    for i in range(19):
        score_dir[jours[i]]=f[i]
    score_dir_sorted= {k: v for k, v in sorted(score_dir.items(), key=lambda item: item[1],reverse=True)}
    #print(score_dir_sorted)
    return score_dir_sorted


if __name__ == '__main__':
    abs = 'Asymmetric multiprocessor systems are considered power-efficient multiprocessor architectures. Furthermore, efficient task allocation (partitioning) can achieve more energy efficiency at these asymmetric multiprocessor platforms. This article addresses the problem of energy-aware static partitioning of periodic real-time tasks on asymmetric multiprocessor (multicore) embedded systems. The article formulates the problem according to the Dynamic Voltage and Frequency Scaling (DVFS) model supported by the platform and shows that it is an NP-hard problem. Then, the article outlines optimal reference partitioning techniques for each case of DVFS model with suitable assumptions. Finally, the article proposes modifications to the traditional bin-packing techniques and designs novel techniques taking into account the DVFS model supported by the platform. All algorithms and techniques are simulated and compared. The simulation shows promising results, where the proposed techniques reduced the energy consumption by 75\% compared to traditional methods when DVFS is not supported and by 50\% when per-core DVFS is supported by the platform.'
    keyw = 'multiprocessor systems'
    r = 0.5
    f = main(abs, keyw, r)
    #print(f)
    score_dir_sorted=final(f)
    for (k,v) in score_dir_sorted.items():
        print(k,v)
