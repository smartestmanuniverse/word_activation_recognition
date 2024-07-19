#coding: utf-8

import threading

class Overflow(Exception):
    pass

class Underflow(Exception):
    pass

class RingBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = [None] * capacity
        self.head = 0
        self.tail = 0
        self.count = 0

    def is_full(self):
        return self.count == self.capacity

    def is_empty(self):
        return self.count == 0

    def enqueue(self, item):
        if self.is_full():
            raise Overflow("Ring buffer is full")
        self.buffer[self.tail] = item
        self.tail = (self.tail + 1) % self.capacity
        self.count += 1

    def dequeue(self):
        if self.is_empty():
            raise Underflow("Ring buffer is empty")
        item = self.buffer[self.head]
        self.buffer[self.head] = None
        self.head = (self.head + 1) % self.capacity
        self.count -= 1
        return item

class ConcurrentRingBuffer:
    def __init__(self, capacity):
        self.ring_buffer = RingBuffer(capacity)
        self.lock = threading.Lock()
        self.not_empty = threading.Condition(self.lock)
        self.not_full = threading.Condition(self.lock)

    def enqueue(self, item):
        with self.not_full:
            while self.ring_buffer.is_full():
                self.not_full.wait()
            self.ring_buffer.enqueue(item)
            self.not_empty.notify()

    def dequeue(self):
        with self.not_empty:
            while self.ring_buffer.is_empty():
                self.not_empty.wait()
            item = self.ring_buffer.dequeue()
            self.not_full.notify()
            return item