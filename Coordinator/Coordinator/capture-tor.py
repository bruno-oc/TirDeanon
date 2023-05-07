import os
import time

SERVER = '192.168.1.77'
TOPO = '0tir'
TYPE = 'peers'
DURATION = 300

for i in range(0, 100):
    print("\nRun#" + str(i))
    os.system(f'START /B "TShark" tshark -i "Adapter for loopback traffic capture" -w "{TOPO}\\{TYPE}\\cap{i}-A-upstream.pcapng" -a duration:{DURATION} -f "dst port 9150"')
    os.system(f'START /B "TShark" tshark -i "Adapter for loopback traffic capture" -w "{TOPO}\\{TYPE}\\cap{i}-A-downstream.pcapng" -a duration:{DURATION} -f "src port 9150"')
    os.system(f'START /B "TShark" tshark -i Ethernet -w "{TOPO}\\{TYPE}\\cap{i}-B-upstream.pcapng" -a duration:{DURATION} -f "src host {SERVER} and not arp"')
    os.system(f'START /B "TShark" tshark -i Ethernet -w "{TOPO}\\{TYPE}\\cap{i}-B-downstream.pcapng" -a duration:{DURATION} -f "dst host {SERVER} and not arp"')
    time.sleep(2)
    os.system('START /B "Client" java -jar TorPeer.jar')
    time.sleep(DURATION + 5)
