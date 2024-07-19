#coding: utf-8
import threading
class Overflow(Exception):
    pass

class Underflow(Exception):
    pass

class RingBuffer:
    def __init__(self, buf):
        self._buf = buf
        self._r = 0
        self._size = 0

    def __len__(self):
        return len(self._buf)

    @property
    def read_size(self):
        return self._size

    @property
    def write_size(self):
        return len(self) - self.read_size

    def read_only(self, buf):
        size = len(buf)
        if size == 0:
            return
        if size > self.read_size:
            raise Underflow
        f = self._r
        l = (f + size) % len(self)
        if f < l:
            buf[:] = self._buf[f:l]
        else:
            n = len(self) - f
            buf[:n] = self._buf[f:]
            buf[n:] = self._buf[:l]

    def remove_only(self, size):
        if size < 0:
            raise ValueError("'size' must be a non-negative number")
        if size > self.read_size:
            raise Underflow
        self._r = (self._r + size) % len(self)
        self._size -= size

    def read(self, buf):
        self.read_only(buf)
        self.remove_only(len(buf))

    def write(self, buf):
        size = len(buf)
        if size == 0:
            return
        if size > self.write_size:
            raise Overflow
        f = (self._r + self._size) % len(self)
        l = (f + size) % len(self)
        if f < l:
            self._buf[f:l] = buf
        else:
            n = len(self) - f
            self._buf[f:] = buf[:n]
            self._buf[:l] = buf[n:]
        self._size += size

class ConcurrentRingBuffer:
    def __init__(self, buf):
        self._rb = RingBuffer(buf)
        self._lock = threading.Lock()
        self._overflow = threading.Condition(self._lock)
        self._underflow = threading.Condition(self._lock)

    def write(self, buf, block=True, timeout=None):
        if len(buf) > len(self._rb):
            raise ValueError("'buf' is too big")
        with self._lock:
            if block and not self._overflow.wait_for(lambda: len(buf) <= self._rb.write_size, timeout):
                raise Overflow
            self._rb.write(buf)
            self._underflow.notify()

    def read(self, buf, remove_size=None, block=True, timeout=None):
        if len(buf) > len(self._rb):
            raise ValueError("'buf' is too big")
        if remove_size is not None and (remove_size < 0 or remove_size > len(buf)):
            raise ValueError("'remove_size' must be non-negative and not exceed 'len(buf)'")
        with self._lock:
            if block and not self._underflow.wait_for(lambda: len(buf) <= self._rb.read_size, timeout):
                raise Underflow
            self._rb.read_only(buf)
            self._rb.remove_only(len(buf) if remove_size is None else remove_size)
            self._overflow.notify()