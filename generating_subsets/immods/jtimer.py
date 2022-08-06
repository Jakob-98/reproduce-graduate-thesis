import time

class Timer:
   def __init__(self, printupdates = True):
    self.starttime = time.time()
    self.deltatime = time.time()
    self.printupdates = printupdates #and config.printTimer
    #self.printtotal = config.printTimer

   def updatetime(self, updatemsg: str):
    if not self.printupdates: return
    prevtime = self.deltatime
    self.deltatime = time.time()
    print(updatemsg, '%.2f' % float(1000*(self.deltatime - prevtime)), 'ms')
   
   def totalTime(self, updatemsg: str):
    if not self.printtotal: return
    print(updatemsg, '%.2f' % float(1000*(time.time() - self.starttime)), 'ms')