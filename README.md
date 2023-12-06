# Maza-Runner
The Maza Program in Python

## Maze Solver
### Process maze image
You need draw the start point with the green color and the end point with the red color. Also seal the margin of the maze with the black color.
### Solve and output image
```
$  python solve.py <maze-image-in> <search-method>(default is BFS)

for example, python solve.py mazes\maze-make-it-difficult3.png BFS  
```

### Solve and output gif & avi
```
$ ./solve_maze.sh mazes/maze_000.jpg method(default is BFS)
for example, ./solve_maze.sh mazes\maze-make-it-difficult3.png BFS
```

You can get the dynamic results in method-image-name/out dir
### Installation
```
$ pip install -r requirements.txt
```

### Dependencies:
Need Python 3.6 or later.
### System
OPTIONAL: package `ffmpeg` for generating mp4 and gif.
If you just want to output an image, these are not needed.

### PS.
DFS are not the shortest path.
