from mininet.topo import Topo
from mininet.link import TCLink
from mininet.node import OVSSwitch

from .config import TopologyConfig

class SDWANTopo(Topo):
    def __init__(self, cfg: TopologyConfig = TopologyConfig(), **opts):
        super().__init__(**opts)
        self.cfg = cfg
        self._build()

    def _build(self):
        h1 = self.addHost('h1')
        h2 = self.addHost('h2')
        s1 = self.addSwitch('s1', stp=True, cls=OVSSwitch)
        s3 = self.addSwitch('s3', stp=True, cls=OVSSwitch)

        # access links
        self.addLink(h1, s1, bw=self.cfg.host_link.bw_mbps, delay=f"{self.cfg.host_link.delay_ms}ms",
                     loss=self.cfg.host_link.loss_pct, jitter=f"{self.cfg.host_link.jitter_ms}ms",
                     max_queue_size=self.cfg.host_link.max_queue, use_htb=True,
                     intfName1='h1-s1', intfName2='s1-h1')

        self.addLink(h2, s3, bw=self.cfg.host_link.bw_mbps, delay=f"{self.cfg.host_link.delay_ms}ms",
                     loss=self.cfg.host_link.loss_pct, jitter=f"{self.cfg.host_link.jitter_ms}ms",
                     max_queue_size=self.cfg.host_link.max_queue, use_htb=True,
                     intfName1='h2-s3', intfName2='s3-h2')

        # LTE tunnel (eth1)
        self.addLink(s1, s3, cls=TCLink,
            bw=self.cfg.lte.bw_mbps, delay=f"{self.cfg.lte.delay_ms}ms",
            loss=self.cfg.lte.loss_pct, jitter=f"{self.cfg.lte.jitter_ms}ms",
            max_queue_size=self.cfg.lte.max_queue, use_htb=True,
            intfName1='s1-eth1', intfName2='s3-eth1')

        # SAT tunnel (eth2)
        self.addLink(s1, s3, cls=TCLink,
            bw=self.cfg.sat.bw_mbps, delay=f"{self.cfg.sat.delay_ms}ms",
            loss=self.cfg.sat.loss_pct, jitter=f"{self.cfg.sat.jitter_ms}ms",
            max_queue_size=self.cfg.sat.max_queue, use_htb=True,
            intfName1='s1-eth2', intfName2='s3-eth2')
