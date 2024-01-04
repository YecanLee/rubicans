"""
This file works for toy example for testing different schedulers
"""

# IDDPM Scheduler
import numpy as np

a = np.linspace(0, 999, 10)
print(a.round()[::-1])


b = np.arange(0, 9)*(1000//9)
print(b.round()[::-1])

c = np.arange(0, 1000, 1000//9)
print(c.round()[::-1])
