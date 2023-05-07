import os
import time

CLIENT = '192.168.1.73'
TIR = '54.36.163.65'
SERVER = '192.168.1.77'
TOPO = '3tir'
TYPE = 'client'
DURATION = 300

for i in range(0, 250):
    print("\nRun#" + str(i))
    os.system(f'START /B "TShark" tshark -i Ethernet -w "{TOPO}\\{TYPE}\\cap{i}-A-upstream.pcapng" -a duration:{DURATION} -f "src host {CLIENT} and dst host {TIR} and not port 22"')
    os.system(f'START /B "TShark" tshark -i Ethernet -w "{TOPO}\\{TYPE}\\cap{i}-A-downstream.pcapng" -a duration:{DURATION} -f "dst host {CLIENT} and src host {TIR} and not port 22"')
    os.system(f'START /B "TShark" tshark -i Ethernet -w "{TOPO}\\{TYPE}\\cap{i}-B-upstream.pcapng" -a duration:{DURATION} -f "src host {SERVER}"')
    os.system(f'START /B "TShark" tshark -i Ethernet -w "{TOPO}\\{TYPE}\\cap{i}-B-downstream.pcapng" -a duration:{DURATION} -f "dst host {SERVER}"')
    time.sleep(5)
    os.system('START /B "Client" java -jar Client.jar')
    time.sleep(DURATION + 5)
