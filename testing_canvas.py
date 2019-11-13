#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 16:16:54 2019

@author: aramos
"""

import matplotlib.pyplot as plt
import numpy as np
import time
from IPython import display

plt.close('Canva test')
figure_test_canvas = plt.figure('Canva test')

ax_test_canvas = figure_test_canvas.add_subplot(1,1,1)

points = []
for period in range(10):
    time.sleep(1)
    print(period)
    points.append(period)
    ax_test_canvas.clear()
    ax_test_canvas.plot(points, 'x-')
    display.display(points)
    figure_test_canvas.canvas.draw()
    plt.pause(0.01)