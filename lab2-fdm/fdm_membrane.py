
from __future__ import division
from __future__ import print_function

import numpy as np
import math
import sys

from mpi4py import MPI

np.set_printoptions(threshold=np.nan)

F =20
H =0.1
NUMBER_OF_ITERATION = 1000

# Command line parameters
ROWS = 0
COLS = 0
PARTITIONED_ROWS = 0 
PARTITIONED_COLS = 0
NUMBER_OF_CPU  = 0
CPU_GRID = (0,0)


def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))

def unblockshaped(arr, h, w):
    """
    Return an array of shape (h, w) where
    h * w = arr.size

    If arr is of shape (n, nrows, ncols), n sublocks of shape (nrows, ncols),
    then the returned array preserves the "physical" layout of the sublocks.
    """
    n, nrows, ncols = arr.shape
    return (arr.reshape(h//nrows, -1, nrows, ncols)
               .swapaxes(1,2)
               .reshape(h, w))

def pprint(str="", end="\n", comm=MPI.COMM_WORLD):
    """Print for MPI parallel programs: Only rank 0 prints *str*."""
    if comm.rank == 0:
        print(str+end, end=' ')

def is_on_top_border(index, number_of_cols, number_of_rows):
    return index >= 0 and index < number_of_cols

def is_on_bottom_border(index, number_of_cols, number_of_rows):
    return index >= ((number_of_rows - 1) * number_of_cols) and index < number_of_cols * number_of_rows

def is_on_left_border(index, number_of_cols, number_of_rows):
    return index % number_of_cols == 0

def is_on_right_border(index, number_of_cols, number_of_rows):
    return (index - (number_of_cols - 1)) % number_of_cols == 0

def get_top_neighbour(index, number_of_cols, number_of_rows):
    if is_on_top_border(index, number_of_cols, number_of_rows):
        return None
    return index - number_of_cols

def get_bottom_neighbour    (index, number_of_cols, number_of_rows):
    if is_on_bottom_border(index, number_of_cols, number_of_rows):
        return None
    return index + number_of_cols


def get_left_neighbour(index, number_of_cols, number_of_rows):
    if is_on_left_border(index, number_of_cols, number_of_rows):
        return None
    return index - 1

def get_right_neighbour(index, number_of_cols, number_of_rows):
    if is_on_right_border(index, number_of_cols, number_of_rows):
        return None
    return index + 1

dictionary_of_functions_getting_neighbour_index = {
'left' : (lambda rank, number_of_cols, number_of_rows: get_left_neighbour(rank, number_of_cols, number_of_rows)),
'right' : (lambda rank, number_of_cols, number_of_rows: get_right_neighbour(rank, number_of_cols, number_of_rows)),
'top' : (lambda rank, number_of_cols, number_of_rows: get_top_neighbour(rank, number_of_cols, number_of_rows)),
'bottom' : (lambda rank, number_of_cols, number_of_rows: get_bottom_neighbour(rank, number_of_cols, number_of_rows))
}

dictionary_of_functions_extracting_ghost_points_from_local_array = {
    'left' :(lambda array: np.array(reshaped_local_input[:,0])),
    'right' :(lambda array: np.array(reshaped_local_input[:,-1])),
    'top' :(lambda array: np.array(np.array(reshaped_local_input[0,:]))),
    'bottom' :(lambda array: np.array(np.array(reshaped_local_input[-1,:])))
}

def solve_using_jaccobi(negihbours, blocks_to_update, reshaped_local_input):
    negihbours_id = {k: dictionary_of_functions_getting_neighbour_index[k](comm.rank, CPU_GRID[1],CPU_GRID[0]) for k in negihbours}

    for i in range(NUMBER_OF_ITERATION):
        old_reshaped_local_input = np.copy(reshaped_local_input)
        neighbour_ghost_points = {}
        requests = []

        for neighbour in negihbours:
            if neighbour == 'left' or neighbour == 'right':
                neighbour_ghost_points[neighbour] = np.empty(PARTITIONED_ROWS, dtype=np.float64)
            else:
                neighbour_ghost_points[neighbour] = np.empty(PARTITIONED_COLS, dtype=np.float64)
 
        for neighbour in negihbours:
            requests.append(comm.Isend([dictionary_of_functions_extracting_ghost_points_from_local_array[neighbour](reshaped_local_input), MPI.DOUBLE], dest=negihbours_id[neighbour]))
        for neighbour in negihbours:
            requests.append(comm.Irecv([neighbour_ghost_points[neighbour], MPI.DOUBLE], source=negihbours_id[neighbour]))

        MPI.Request.waitall(requests)

        #interior
        for row in range(1, PARTITIONED_ROWS - 1):
            for col in range(1, PARTITIONED_COLS - 1):
                reshaped_local_input[row][col] = calculate_next_approximation(
                    old_reshaped_local_input[row][col - 1],
                    old_reshaped_local_input[row][col + 1],
                    old_reshaped_local_input[row + 1][col],
                    old_reshaped_local_input[row - 1][col])
        
        if 'left-border' in blocks_to_update:
            for row in range(1, PARTITIONED_ROWS - 1):
                reshaped_local_input[row][0] = calculate_next_approximation(
                    neighbour_ghost_points['left'][row],
                    old_reshaped_local_input[row][1],
                    old_reshaped_local_input[row -1][0],
                    old_reshaped_local_input[row + 1][0])
        if 'right-border' in blocks_to_update:
            for row in range(1, PARTITIONED_ROWS - 1):
                reshaped_local_input[row][PARTITIONED_COLS - 1] = calculate_next_approximation(
                    old_reshaped_local_input[row][PARTITIONED_COLS - 2],
                    neighbour_ghost_points['right'][row],
                    old_reshaped_local_input[row -1][PARTITIONED_COLS - 1],
                    old_reshaped_local_input[row + 1][PARTITIONED_COLS - 1])
        if 'top-border' in blocks_to_update:
            for col in range(1, PARTITIONED_COLS - 1):
                reshaped_local_input[0][col] = calculate_next_approximation(
                old_reshaped_local_input[0][col - 1],
                old_reshaped_local_input[0][col + 1],
                neighbour_ghost_points['top'][col],
                old_reshaped_local_input[1][col])
        if 'bottom-border' in blocks_to_update:
            for col in range(1, PARTITIONED_COLS - 1):
                reshaped_local_input[PARTITIONED_ROWS - 1][col] = calculate_next_approximation(
                    old_reshaped_local_input[PARTITIONED_ROWS - 1][col - 1],
                    old_reshaped_local_input[PARTITIONED_ROWS - 1][col + 1],
                    old_reshaped_local_input[PARTITIONED_ROWS - 2][col],
                    neighbour_ghost_points['bottom'][col])
        if 'top-left-corner' in blocks_to_update:
            reshaped_local_input[0][0] = calculate_next_approximation(
                neighbour_ghost_points['left'][0],
                old_reshaped_local_input[0][1],
                neighbour_ghost_points['top'][0],
                old_reshaped_local_input[1][0])
        if 'top-right-corner' in blocks_to_update:
            reshaped_local_input[0][PARTITIONED_COLS - 1] = calculate_next_approximation(
                old_reshaped_local_input[0][PARTITIONED_COLS - 2],
                neighbour_ghost_points['right'][0],
                neighbour_ghost_points['top'][PARTITIONED_COLS -1],
                old_reshaped_local_input[1][PARTITIONED_COLS - 1])
        if 'bottom-left-corner' in blocks_to_update:
            reshaped_local_input[PARTITIONED_ROWS - 1][0] = calculate_next_approximation(
                neighbour_ghost_points['left'][PARTITIONED_ROWS -1],
                old_reshaped_local_input[PARTITIONED_ROWS - 1][1],
                old_reshaped_local_input[PARTITIONED_ROWS - 2][0],
                neighbour_ghost_points['bottom'][0])
        if 'bottom-right-corner' in blocks_to_update:
            reshaped_local_input[PARTITIONED_ROWS - 1][PARTITIONED_COLS - 1] = calculate_next_approximation(
                old_reshaped_local_input[PARTITIONED_ROWS - 1][PARTITIONED_COLS -2],
                neighbour_ghost_points['right'][PARTITIONED_ROWS - 1],
                old_reshaped_local_input[PARTITIONED_ROWS - 2][PARTITIONED_COLS -1],
                neighbour_ghost_points['bottom'][PARTITIONED_COLS - 1])

def calculate_next_approximation(left_neighbour, right_neighbour, top_neighbour, bottom_neighbour):
    return 0.25 * (
        left_neighbour +
        right_neighbour +
        bottom_neighbour +
        top_neighbour +
        H*H*F)

comm = MPI.COMM_WORLD

NUMBER_OF_CPU  = comm.size

ROWS = int(sys.argv[1])
COLS = int(sys.argv[2])
PARTITIONED_ROWS = int(sys.argv[3]) 
PARTITIONED_COLS = int(sys.argv[4])
CPU_GRID = (int(sys.argv[5]), int(sys.argv[6]))

if comm.rank == 0:
    input = np.zeros(ROWS * COLS, dtype=np.float64).reshape((ROWS,COLS))
    input = blockshaped(input, PARTITIONED_ROWS, PARTITIONED_COLS)
else:
    input = np.empty(ROWS * COLS, dtype=np.float64)

local_input = np.empty(PARTITIONED_ROWS * PARTITIONED_COLS, dtype=np.float64)

# Scatter input data into my_local arrays
comm.Scatter( [input, MPI.DOUBLE], [local_input, MPI.DOUBLE] )

reshaped_local_input = local_input.reshape(PARTITIONED_ROWS, PARTITIONED_COLS)
output = np.empty(ROWS * COLS, dtype=np.float64)
for r in range(comm.size):
    if comm.size == 1:
        directions_of_neighbouring_processes = []
        blocks_to_update = []
        solve_using_jaccobi(directions_of_neighbouring_processes, blocks_to_update, reshaped_local_input)
    elif comm.rank == r:
        if is_on_left_border(comm.rank, CPU_GRID[1],CPU_GRID[0]) and is_on_right_border(comm.rank, CPU_GRID[1],CPU_GRID[0]) and is_on_top_border(comm.rank, CPU_GRID[1],CPU_GRID[0]):
            directions_of_neighbouring_processes = ['bottom']
            blocks_to_update = ['bottom-border']
            solve_using_jaccobi(directions_of_neighbouring_processes, blocks_to_update, reshaped_local_input)            
        elif is_on_left_border(comm.rank, CPU_GRID[1],CPU_GRID[0]) and is_on_right_border(comm.rank, CPU_GRID[1],CPU_GRID[0]) and is_on_bottom_border(comm.rank, CPU_GRID[1],CPU_GRID[0]):
            directions_of_neighbouring_processes = ['top']
            blocks_to_update = ['top-border']
            solve_using_jaccobi(directions_of_neighbouring_processes, blocks_to_update, reshaped_local_input)
        elif is_on_left_border(comm.rank, CPU_GRID[1],CPU_GRID[0]) and is_on_right_border(comm.rank, CPU_GRID[1],CPU_GRID[0]):
            directions_of_neighbouring_processes = ['top', 'bottom']
            blocks_to_update = ['top-border', 'bottom-border']
            solve_using_jaccobi(directions_of_neighbouring_processes, blocks_to_update, reshaped_local_input)
        elif (is_on_top_border(comm.rank, CPU_GRID[1],CPU_GRID[0]) and is_on_left_border(comm.rank, CPU_GRID[1],CPU_GRID[0])):
            directions_of_neighbouring_processes = ['right', 'bottom']
            blocks_to_update = ['right-border', 'bottom-border', 'bottom-right-corner']
            solve_using_jaccobi(directions_of_neighbouring_processes, blocks_to_update, reshaped_local_input)

        elif(is_on_top_border(comm.rank, CPU_GRID[1],CPU_GRID[0]) and is_on_right_border(comm.rank, CPU_GRID[1],CPU_GRID[0])):
            directions_of_neighbouring_processes = ['left', 'bottom']
            blocks_to_update = ['left-border', 'bottom-border', 'bottom-left-corner']
            solve_using_jaccobi(directions_of_neighbouring_processes, blocks_to_update, reshaped_local_input)

        elif(is_on_bottom_border(comm.rank, CPU_GRID[1],CPU_GRID[0]) and is_on_left_border(comm.rank, CPU_GRID[1],CPU_GRID[0])):
            directions_of_neighbouring_processes = ['right', 'top']
            blocks_to_update = ['right-border', 'top-border', 'top-right-corner']
            solve_using_jaccobi(directions_of_neighbouring_processes, blocks_to_update, reshaped_local_input)

        elif(is_on_bottom_border(comm.rank, CPU_GRID[1],CPU_GRID[0]) and is_on_right_border(comm.rank, CPU_GRID[1],CPU_GRID[0])):
            directions_of_neighbouring_processes = ['left', 'top']
            blocks_to_update = ['left-border', 'top-border', 'top-left-corner']
            solve_using_jaccobi(directions_of_neighbouring_processes, blocks_to_update, reshaped_local_input)

        elif(is_on_top_border(comm.rank, CPU_GRID[1],CPU_GRID[0])):
            directions_of_neighbouring_processes = ['left', 'right', 'bottom']
            blocks_to_update = ['left-border', 'right-border', 'bottom-border', 'bottom-right-corner', 'bottom-left-corner']
            solve_using_jaccobi(directions_of_neighbouring_processes, blocks_to_update, reshaped_local_input)

        elif(is_on_bottom_border(comm.rank, CPU_GRID[1],CPU_GRID[0])):
            directions_of_neighbouring_processes = ['left', 'right', 'top']
            blocks_to_update = ['left-border', 'right-border', 'top-border', 'top-right-corner', 'top-left-corner']
            solve_using_jaccobi(directions_of_neighbouring_processes, blocks_to_update, reshaped_local_input)

        elif(is_on_right_border(comm.rank, CPU_GRID[1],CPU_GRID[0])):
            directions_of_neighbouring_processes = ['left', 'top', 'bottom']
            blocks_to_update = ['left-border', 'top-border', 'bottom-border', 'top-left-corner', 'bottom-left-corner']
            solve_using_jaccobi(directions_of_neighbouring_processes, blocks_to_update, reshaped_local_input)

        elif(is_on_left_border(comm.rank, CPU_GRID[1],CPU_GRID[0])):
            directions_of_neighbouring_processes = ['right', 'top', 'bottom']
            blocks_to_update = ['right-border', 'top-border', 'bottom-border', 'top-right-corner', 'bottom-right-corner']
            solve_using_jaccobi(directions_of_neighbouring_processes, blocks_to_update, reshaped_local_input)

        else:
            directions_of_neighbouring_processes = ['left','right', 'top', 'bottom']
            blocks_to_update = ['left-border', 'right-border', 'top-border', 'bottom-border', 'top-left-corner', 'top-right-corner', 'bottom-left-corner', 'bottom-right-corner' ]
            solve_using_jaccobi(directions_of_neighbouring_processes, blocks_to_update, reshaped_local_input)

comm.Gather( [reshaped_local_input.flatten(), MPI.DOUBLE], [output, MPI.DOUBLE], root=0)

if comm.rank==0:
    linearized_local_blocks = np.split(output, NUMBER_OF_CPU)
    reshaped_local_blocks = [x.reshape(PARTITIONED_ROWS, PARTITIONED_COLS) for x in linearized_local_blocks]
    merged_blocks = unblockshaped(np.asarray(reshaped_local_blocks), ROWS, COLS)
    np.savetxt(sys.stdout, merged_blocks)
