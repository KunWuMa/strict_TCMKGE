"""Cross-process locks prevent two local queues writing the same checkpoints."""
import os


class ExclusiveLock:
    def __init__(self,path):self.path=path;self.handle=None
    def __enter__(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.handle=self.path.open('a+b')
        self.handle.seek(0,os.SEEK_END)
        if self.handle.tell()==0:self.handle.write(b'0');self.handle.flush()
        self.handle.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(self.handle.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError:
            self.handle.close();self.handle=None
            raise RuntimeError('Another experiment queue holds '+str(self.path))
        return self
    def __exit__(self,*args):
        if self.handle is not None:self.handle.close()
