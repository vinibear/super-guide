import jieba.analyse
sentence=("Intellectual Property (IP) reuse is a well known practice in chip design processes. "
          "Nowadays, network-on-chips (NoCs) are increasingly used as IP and sold by various vendors "
          "to be integrated in a multiprocessor system-on-chip (MPSoC). "
          "However, IP reuse exposes the design to IP theft, and an attacker can launch IP stealing "
          "attacks against NoC IPs. With the growing adoption of MPSoC, such attacks can result in "
          "huge financial losses. In this article, we propose four NoC IP protection techniques "
          "using fingerprint embedding: ON-OFF router-based fingerprinting (ORF), ON-OFF link-based "
          "fingerprinting (OLF), Router delay-based fingerprinting (RTDF), and Row delay-based "
          "fingerprinting (RWDF). ORF and OLF techniques use patterns of ON-OFF routers and links, "
          "respectively, while RTDF and RWDF techniques use router delays to embed fingerprints. "
          "We show that all of our proposed techniques require much less hardware overhead compared "
          "to an existing NoC IP security solution (square spiral routing) and also provide better "
          "security from removal and masking attacks. In particular, our proposed techniques require "
          "between 40.75% and 48.43% less router area compared to the existing solution. We also show "
          "that our solutions do not affect the normal packet latency and hence do not degrade the NoC performance.")
r=jieba.analyse.extract_tags(sentence, topK=20, withWeight=False, allowPOS=())
print(r)