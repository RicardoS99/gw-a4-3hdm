
import os
from progress.bar import Bar
import time
import multiprocessing as mp


class MyProcessManager():
    def __init__(self, target, queue, nprocesses = os.cpu_count(), args = None, timeout=86400, lag=0.0):
        self.target = target
        self.queue = queue
        self.args = args
        if nprocesses>0:
            self.nprocesses = nprocesses
        else:
            self.nprocesses = os.cpu_count()

        if timeout>0:
            self.timeout = timeout
        else:
            self.timeout = 86400
        
        if lag>=0.0:
            self.lag = lag
        else:
            self.lag = 0.0


    def run(self):
        processes = []
        timers = {}
        nmax = self.queue.qsize()
        rproc = nmax

        bar = Bar('Processing', max = nmax, suffix='%(index)d/%(max)d (%(percent).1f%%) - Elapsed Time: %(elapsed)ds - ETA: %(eta)ds')
        for k in range(self.nprocesses):
            if rproc>0:
                processes.append(mp.Process(target=self.target, args=self.args))
                processes[-1].start()
                timers[processes[-1].pid] = time.time()
                rproc-=1

        while len(processes)>0:
            for proc in processes:
                if not proc.is_alive():
                    proc.join()
                    del timers[proc.pid]
                    proc.close()
                    processes.remove(proc)
                    
                    if rproc>0:
                        processes.append(mp.Process(target=self.target, args=self.args))
                        processes[-1].start()
                        timers[processes[-1].pid] = time.time()
                        rproc-=1
                    bar.next()
                elif time.time() - timers[proc.pid] > self.timeout:
                    proc.terminate()
                    proc.join()
                    del timers[proc.pid]
                    proc.close()
                    processes.remove(proc)                    
                    if rproc>0:
                        processes.append(mp.Process(target=self.target, args=self.args))
                        processes[-1].start()
                        timers[processes[-1].pid] = time.time()
                        rproc-=1
                    bar.next()
            time.sleep(self.lag)
        bar.finish()