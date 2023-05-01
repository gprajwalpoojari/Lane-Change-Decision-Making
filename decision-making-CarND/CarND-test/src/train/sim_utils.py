import socket
import subprocess
import time
import psutil
import pyautogui
import os
from pynput.mouse import Listener
from multiprocessing import Pool

def connect(ser):
    conn, addr = ser.accept()
    print('Connected by', addr)
    return conn

def open_ter(loc):
    os.system("gnome-terminal -e 'bash -c \"cd " + loc + " && ./path_planning; exec bash\"'")
    time.sleep(1)

def setup_server(host='127.0.0.1', port=1234):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((host, port))
    server.listen(1)
    return server

def launch_simulator(server, location="/home/prajwal/Autonomous-Driving/decision-making-CarND/CarND-test/build"):
    pool = Pool(processes=2)
    result = []
    result.append(pool.apply_async(connect, (server,)))
    pool.apply_async(open_ter, (location,))
    pool.close()
    pool.join()
    conn = result[0].get()
    sim = subprocess.Popen('/home/prajwal/Autonomous-Driving/decision-making-CarND/term3_sim_linux/term3_sim.x86_64')
    while not Listener(on_click=_on_click_):
        pass
    time.sleep(2)
    pyautogui.click(x=1913, y=1426, button='left')
    time.sleep(6)
    pyautogui.click(x=1708, y=1711, button='left')
    return conn, sim

def close_all(sim):
    if sim.poll() is None:
        sim.terminate()
        sim.wait()
    time.sleep(2)
    kill_terminal()

def kill_terminal():
    pids = psutil.pids()
    for pid in pids:
        p = psutil.Process(pid)
        if p.name() == "gnome-terminal-server":
            os.kill(pid, 9)

def _on_click_(x, y, button, pressed):
    return pressed
