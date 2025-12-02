#!/usr/bin/env python3
"""
Simple SDN Topology with Multiple Hosts and Switches
"""

from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import RemoteController, OVSKernelSwitch
from mininet.cli import CLI
from mininet.log import setLogLevel, info
from mininet.link import TCLink

class CustomTopology(Topo):
    """
    Custom topology with 3 switches and 6 hosts
    
    Topology:
              h1    h2
               \    /
                s1
                 |
                s2
               /  \
              s3   s4
             / \   / \
            h3 h4 h5 h6
    """
    
    def build(self):
        # Add switches
        s1 = self.addSwitch('s1', protocols='OpenFlow13')
        s2 = self.addSwitch('s2', protocols='OpenFlow13')
        s3 = self.addSwitch('s3', protocols='OpenFlow13')
        s4 = self.addSwitch('s4', protocols='OpenFlow13')
        
        # Add hosts
        h1 = self.addHost('h1', ip='10.0.0.1/24', mac='00:00:00:00:00:01')
        h2 = self.addHost('h2', ip='10.0.0.2/24', mac='00:00:00:00:00:02')
        h3 = self.addHost('h3', ip='10.0.0.3/24', mac='00:00:00:00:00:03')
        h4 = self.addHost('h4', ip='10.0.0.4/24', mac='00:00:00:00:00:04')
        h5 = self.addHost('h5', ip='10.0.0.5/24', mac='00:00:00:00:00:05')
        h6 = self.addHost('h6', ip='10.0.0.6/24', mac='00:00:00:00:00:06')
        
        # Add links between switches with bandwidth constraints
        self.addLink(s1, s2, cls=TCLink, bw=100)
        self.addLink(s2, s3, cls=TCLink, bw=50)
        self.addLink(s2, s4, cls=TCLink, bw=50)
        
        # Add links between hosts and switches
        self.addLink(h1, s1, cls=TCLink, bw=10)
        self.addLink(h2, s1, cls=TCLink, bw=10)
        self.addLink(h3, s3, cls=TCLink, bw=10)
        self.addLink(h4, s3, cls=TCLink, bw=10)
        self.addLink(h5, s4, cls=TCLink, bw=10)
        self.addLink(h6, s4, cls=TCLink, bw=10)

def run_topology():
    """Start the network with remote controller"""
    topo = CustomTopology()
    
    # Create network with remote controller (Ryu)
    net = Mininet(
        topo=topo,
        controller=lambda name: RemoteController(
            name, ip='127.0.0.1', port=6653
        ),
        switch=OVSKernelSwitch,
        link=TCLink,
        autoSetMacs=True
    )
    
    info('*** Starting network\n')
    net.start()
    
    info('*** Testing connectivity\n')
    net.pingAll()
    
    info('*** Running CLI\n')
    CLI(net)
    
    info('*** Stopping network\n')
    net.stop()

if __name__ == '__main__':
    setLogLevel('info')
    run_topology()