#!/usr/bin/env python3

"""
    Maze Solver

    Usage:
        python solve.py <maze-image-in> method(default is BFS)

    Output:
        An image of the original maze with the solution path drawn in.

    Note:
        This program relies on colors.
        For example, assumes explorable space is WHITE and the maze is BLACK.
"""

import os
import sys
import math
import time
import logging
from PIL import Image
import urllib.parse
from queue import PriorityQueue

logging.basicConfig(level=logging.INFO, format="[%(levelname)s]: %(asctime)-15s %(message)s")

method = 'BFS'

class Solver:
    """
    file_in = Input maze image filename.
    image   = Maze image object.
    pixels  = Maze image pixel array.
    """
    def __init__(self, maze):

       
        # Colors.
        self.COLOR_MAP = {
            (0,255,0): 'GREEN',
            (255,0,0): 'RED',
            (0,0,255): 'BLUE',
            (255,255,255): 'WHITE',
            (0,0,0): 'BLACK'
        }
        self.COLOR_RED = (255,0,0)
        self.COLOR_GREEN = (0,255,0)
        self.COLOR_BLUE = (0,0,255)
        self.COLOR_WHITE = (255,255,255)
        self.COLOR_BLACK = (0,0,0)
        self.START_COLOR = self.COLOR_GREEN
        self.END_COLOR = self.COLOR_RED
        self.FRONTIER_COLOR = self.COLOR_GREEN
        self.memoized_color_map = {}

        dirname = urllib.parse.quote_plus(method + '-' + os.path.basename(maze))
        # 
        os.system('mkdir -p ' + dirname)
        os.system('mkdir -p ' + os.path.join(dirname, 'out'))
        os.system('mkdir -p ' + os.path.join(dirname, 'tmp'))
        # Output file.
        self.DIR_OUT = os.path.join(dirname, 'out')
        self.file_in = maze
        ext = maze.split('.')[-1]
        self.file_out = os.path.join(self.DIR_OUT, os.path.basename(maze).split('.')[0] + '.' + ext)

        # Output parameters.
        self.SNAPSHOT_FREQ = 20000 # Save an image every SNAPSHOT_FREQ steps.

        # BFS parameters.
        self.tmp_dir = os.path.join(dirname, 'tmp')
        self.iterations = 0

        # Load image.
        self.image = Image.open(self.file_in)
        logging.info("Loaded image '{0}' ({1} = {2} pixels).".format(
            self.file_in, self.image.size, self.image.size[0]*self.image.size[1]))
        self.image = self.image.convert('RGB')
        self.pixels = self.image.load()
        self.START = self._findStart()
        self.END = self._findEnd()
        self._saveImage(self.image, '{0}/start_end.jpg'.format(self.tmp_dir))
        self._cleanImage()
        self._drawSquare(self.START, self.START_COLOR)
        self._drawSquare(self.END, self.END_COLOR)
        self._saveImage(self.image, '{0}/clean.jpg'.format(self.tmp_dir))


    """
    Purify pixels to either pure black or white, except for the start/end pixels.
    """
    def _cleanImage(self):
        logging.info("Cleaning image...")
        x,y = self.image.size
        for i in range(x):
            for j in range(y):
                if (i,j) == self.START:
                    self.pixels[i,j] == self.START_COLOR
                    continue
                if (i,j) == self.END:
                    self.pixels[i,j] == self.END_COLOR
                    continue
                closest_color = self._findClosestColor(self.pixels[i,j])
                for color in [self.COLOR_WHITE, self.COLOR_BLACK]:
                    if closest_color == color: self.pixels[i,j] = color
                for color in [self.START_COLOR, self.END_COLOR]:
                    if closest_color == color: self.pixels[i,j] = self.COLOR_WHITE

    def _findClosestColor(self, color, memoize=False):
        colors = list(self.COLOR_MAP.keys())
        if color in self.memoized_color_map and memoize == True:
            return color
        closest_color = sorted(colors, key=lambda c: distance(c, color))[0]
        if memoize == True: self.memoized_color_map[color] = closest_color
        return closest_color

    def _findColorCenter(self, color):
        found_color = False
        x_min, x_max, y_min, y_max = float('inf'), float('-inf'), float('inf'), float('-inf')
        x,y = self.image.size
        for i in range(x):
            for j in range(y):
                code = self._findClosestColor(self.pixels[i,j])
                if  code == color:
                    found_color = True
                    x_min, y_min = min(x_min, i), min(y_min, j)
                    x_max, y_max = max(x_max, i), max(y_max, j)
        if not found_color:
            return (0,0), False
        
        return (int( x_max), int( y_max)), True

    def _findStart(self):
        logging.info("Finding START point...")
        start, ok = self._findColorCenter(self.START_COLOR)
        if not ok:
           logging.error("Oops, failed to find start point in maze!")
        self._drawSquare(start, self.START_COLOR)
        logging.info(start)
        return start

    def _findEnd(self):
        logging.info("Finding END point...")
        end, ok = self._findColorCenter(self.END_COLOR)
        if not ok:
            logging.error("Oops, failed to find end point in maze!")
        self._drawSquare(end, self.END_COLOR)
        logging.info(end)
        return end

    def solve(self):
        logging.info('Solving...')
        path = None
        if method == 'BFS':
            path = self._BFS(self.START, self.END)
        elif method == 'DFS':
            path = self._DFS(self.START, self.END)
        elif method == 'AStar':
            path = self._AStar(self.START, self.END)
        elif method == 'BI':
            path = self._bi_directional_search(self.START, self.END)
            
        # 
        if path is None:
            logging.error('No path found.')
            self._drawX(self.START)
            self._drawX(self.END)
            self.image.save(self.file_out)
            sys.exit(1)

        # Draw solution path.
        for position in path:
            x,y = position
            self.pixels[x,y] = self.COLOR_RED
        # 
        self._save_path(path)

        self.image.save(self.file_out)
        logging.info("Solution saved as '{0}'.".format(self.file_out))

    def _drawX(self, pos, color=(0,0,255)):
        x,y = pos
        d = 10
        for i in range(-d,d):
            self.pixels[x+i,y] = color
        for j in range(-d,d):
            self.pixels[x,y+j] = color

    def _drawSquare(self, pos, color=(0,0,255)):
        x,y = pos
        d = 1
        for i in range(-d,d):
            for j in range(-d,d):
                self.pixels[x+i,y+j] = color

    def _inBounds(self, dim, x, y):
        mx, my = dim
        if x < 0 or y < 0 or x >= mx or y >= my:
            return False
        return True

    def _isWhite(self, pixels, pos):
        pixels = self.pixels
        i,j = pos
        r,g,b = pixels[i,j]
        th = 240
        if pixels[i,j] == self.COLOR_WHITE or pixels[i,j] == 0 or (r>th and g>th and b>th) \
        or pixels[i,j] == self.END_COLOR or pixels[i,j] == self.START_COLOR:
            return True

    # Left, Down, Right, Up
    def _getNeighbours(self, pos):
        x,y = pos
        return [(x-1,y),(x,y-1),(x+1,y),(x,y+1)]

    def _saveImage(self, img, path):
        img.save(path)
    
    def _save_path(self, path):
        for p in path:
            with open(os.path.join(self.DIR_OUT, 'path.txt'), 'a') as f:
                f.write('{0}\n'.format(str(p)))

    """
    Breadth-first search.
    """
    def _BFS(self, start, end):
        # Copy of maze to hold temporary search state.
        image = self.image.copy()
        pixels = image.load()

        self.iterations = 0
        came_from = dict()
        
        came_from[start] = None
        Q = [start]
        img = 0

        while len(Q) != 0:
            if self.iterations > 0 and self.iterations%self.SNAPSHOT_FREQ==0:
                logging.info("...")
            path = Q.pop(0)
            pos = path
            # if pos[0] == end[0] or pos[1] == end[1]:
                # logging.info(pos)
            if pos == end:
                # Draw solution path.
                path = []
                current = end
                while current != start: 
                    path.append(current)
                    current = came_from[current]
                
                path.append(start)
                path.reverse()

                for position in path:
                    x,y = position
                    pixels[x,y] = self.COLOR_RED
                for i in range(10):
                    self._saveImage(image, '{0}/{1:05d}.jpg'.format(self.tmp_dir, img))
                    img += 1
                logging.info('Found a path after {0} iterations.'.format(self.iterations))
                image.show("Solution Path With BFS")
                return path

            for neighbour in self._getNeighbours(pos):
                x,y = neighbour
                if (x,y) not in came_from and self._inBounds(image.size, x, y) and self._isWhite(pixels, (x,y)):
                    pixels[x,y] = self.FRONTIER_COLOR
                    came_from[neighbour] = pos
                    Q.append(neighbour)
                   
            if self.iterations % self.SNAPSHOT_FREQ == 0:
                self._saveImage(image, '{0}/{1:05d}.jpg'.format(self.tmp_dir, img))
                img += 1
            self.iterations += 1
        print("Returning after ", self.iterations, " iterations.")
        return None
    
    """
    Depth-first search.
    """
    def _DFS(self, start, end):
        image = self.image.copy()
        pixels = image.load()

        self.iterations = 0
        came_from = dict()
        
        came_from[start] = None
        S = [start]
        img = 0

        while len(S) != 0:
            if self.iterations > 0 and self.iterations%self.SNAPSHOT_FREQ==0:
                logging.info("...")
            path = S.pop()
            pos = path
            
            if pos == end:
                # Draw solution path.
                path = []
                current = end
                while current != start: 
                    path.append(current)
                    current = came_from[current]

                path.append(start)
                path.reverse()

                for position in path:
                    x,y = position
                    pixels[x,y] = self.COLOR_RED
                for i in range(10):
                    self._saveImage(image, '{0}/{1:05d}.jpg'.format(self.tmp_dir, img))
                    img += 1
                logging.info('Found a path after {0} iterations.'.format(self.iterations))
                image.show("Solution Path With DFS")
                return path

            for neighbour in self._getNeighbours(pos):
                x,y = neighbour
                if (x,y) not in came_from and self._inBounds(image.size, x, y) and self._isWhite(pixels, (x,y)):
                    pixels[x,y] = self.FRONTIER_COLOR
                    S.append(neighbour)
                    came_from[neighbour] = pos
            if self.iterations % self.SNAPSHOT_FREQ == 0:
                self._saveImage(image, '{0}/{1:05d}.jpg'.format(self.tmp_dir, img))
                img += 1
            self.iterations += 1
        print("Returning after ", self.iterations, " iterations.")
        return None

    """
    A* search.
    """
    def _AStar(self, start, end):
        image = self.image.copy()
        pixels = image.load()

        self.iterations = 0
        came_from = dict()
        
        came_from[start] = None
        
        img = 0
        frontier = PriorityQueue()
        frontier.put((0,start))
        cost_so_far = dict()
        cost_so_far[start] = 0

        while not frontier.empty():
            if self.iterations > 0 and self.iterations%self.SNAPSHOT_FREQ==0:
                logging.info("...")
            
            _, current = frontier.get()
            
            if current == end:
                # Draw solution path.
                path = []
                current = end
                while current != start: 
                    path.append(current)
                    current = came_from[current]

                path.append(start)
                path.reverse()

                for position in path:
                    x,y = position
                    pixels[x,y] = self.COLOR_RED
                
                for i in range(10):
                    self._saveImage(image, '{0}/{1:05d}.jpg'.format(self.tmp_dir, img))
                    img += 1
                logging.info('Found a path after {0} iterations.'.format(self.iterations))
                image.show("Solution Path With A*")
                return path
            
            for next in self._getNeighbours(current):
                x,y = next
                new_cost = cost_so_far[current] + 1
                if (next not in came_from or new_cost < cost_so_far[next]) and self._inBounds(image.size, x, y) and self._isWhite(pixels, (x,y)):
                    pixels[x,y] = self.FRONTIER_COLOR
                    
                    cost_so_far[next] = new_cost
                    priority = new_cost + self._heuristic(end, next)
                    frontier.put((priority, next))
                    came_from[next] = current
                    
            if self.iterations % self.SNAPSHOT_FREQ == 0:
                self._saveImage(image, '{0}/{1:05d}.jpg'.format(self.tmp_dir, img))
                img += 1
            self.iterations += 1
        print("Returning after ", self.iterations, " iterations.")

    def _heuristic(self, pos, end):
        return abs(pos[0] - end[0]) + abs(pos[1] - end[1])            


    def _bi_directional_search(self, start, end):
        # Copy of maze to hold temporary search state.
        image = self.image.copy()
        pixels = image.load()

        self.iterations = 0
        came_from_s = dict()
        came_from_s[start] = None
        Qs = [start]

        came_from_e = dict()
        came_from_e[end] = None
        Qe = [end]
        img = 0
        
        while len(Qs) != 0 or len(Qe) != 0:
            if self.iterations > 0 and self.iterations%self.SNAPSHOT_FREQ==0:
                logging.info("...")
            

            if self.iterations % 2 == 0 and len(Qs) != 0:
                Q = Qs
                came_from = came_from_s
                came_other = came_from_e
            elif len(Qe) != 0:
                Q = Qe
                came_from = came_from_e
                came_other = came_from_s
            else:
                self.iterations += 1
                continue
            pos = Q.pop(0)
            if pos in came_other:
                path = []
                
                current = pos
                while current != start: 
                    path.append(current)
                    current = came_from_s[current]
                
                path.append(start)
                path.reverse()

                current = pos
                while current != end: 
                    path.append(current)
                    current = came_from_e[current]
                path.append(end)

                for position in path:
                    x,y = position
                    pixels[x,y] = self.COLOR_RED
                for i in range(10):
                    self._saveImage(image, '{0}/{1:05d}.jpg'.format(self.tmp_dir, img))
                    img += 1
                logging.info('Found a path after {0} iterations.'.format(self.iterations))
                image.show("Solution Path With BI")
                return path

            for neighbour in self._getNeighbours(pos):
                x,y = neighbour
                
                if (x,y) not in came_from and self._inBounds(image.size, x, y) and self._isWhite(pixels, (x,y)):
                    pixels[x,y] = self.FRONTIER_COLOR
                    came_from[neighbour] = pos
                    Q.append(neighbour)
                   
            if self.iterations % self.SNAPSHOT_FREQ == 0:
                self._saveImage(image, '{0}/{1:05d}.jpg'.format(self.tmp_dir, img))
                img += 1
            self.iterations += 1
        print("Returning after ", self.iterations, " iterations.")
        return None

        
        

def mean(numbers):
    return int(sum(numbers)) / max(len(numbers), 1)

def distance(c1, c2):
    (r1,g1,b1) = c1
    (r2,g2,b2) = c2
    return math.sqrt((r1 - r2)**2 + (g1 - g2) ** 2 + (b1 - b2) **2)

if __name__ == '__main__':
    method = sys.argv[2]
    solver = Solver(sys.argv[1])
    solver.solve()
