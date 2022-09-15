#! /usr/bin/env python

class Palette:
    def __init__(self, x, y, z, paint_diff=0.03976):
        self.position = (x, y, z)
        # paint_diff is distance between each paint on the palette
        self.paint_diff = paint_diff

class Rag:
    def __init__(self, x, y, z):
        self.position = (x, y, z)

class WaterBowl:
    def __init__(self, x, y, z):
        self.position = (x, y, z)

class PaintBrush:
    def __init__(self):
        self.clean = True

class Canvas:
    def __init__(self, x, y, z, height=0):
        self.height = height
        self.position = (x, y, z)