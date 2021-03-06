import sys
import copy

from mpi4py import MPI
from star import Star
from star_generator import StarGenerator
from star_interactions_acumlator import StarInteractionsAccumlator


def number_of_process_additional_stars(number_of_stars, number_of_processes, process_id):
	if process_id < number_of_stars % number_of_processes:
		return 1
	else:
		return 0

def find_number_of_stars_for_process(number_of_stars, number_of_processes, process_id):
	main_part = number_of_stars / number_of_processes
	main_part += number_of_process_additional_stars(number_of_stars, number_of_processes, process_id)

	return main_part

def compute_interaction_between_stars(own_stars, stars_buffer, accumlator):
	for i in range(len(own_stars)):
		for j in range(len(stars_buffer)):
			#print('Debug:',id, i , j, str(own_stars[i]), str(stars_buffer[j]))
			accumlator[i].update(own_stars[i], stars_buffer[j])

def print_result(id, own_stars, accumlator):
	for i in range(len(own_stars)):
		print(id, str(own_stars[i]), str(accumlator[i]))

if len(sys.argv) != 2:
	exit_msg ='Incorrect number of arguments. Actual: {} expected: {}. \n'.format(len(sys.argv) - 1, 1)
	exit_msg += 'Usage python <PYTHON_SCRIPT_NAME> <NUMBER_OF_STARS>'
	sys.exit(exit_msg)

comm = MPI.COMM_WORLD

MAX_WEIGHT = 100000000000
MIN_WEIGHT = 100000
MAX_COORDINATE = 10000
NUMBER_OF_STARS = int(sys.argv[1])
p = comm.size

id = comm.rank

stars_generator = StarGenerator(MIN_WEIGHT, MAX_WEIGHT, MAX_COORDINATE)

#own_stars = stars_generator.generate_concrete_list_of_stars_for(id)
own_stars = stars_generator.generate_random_list_of_stars(find_number_of_stars_for_process(NUMBER_OF_STARS, p, id))
accumlator = [StarInteractionsAccumlator() for acc in range(find_number_of_stars_for_process(NUMBER_OF_STARS, p, id))]
stars_buffer = own_stars

compute_interaction_between_stars(own_stars, stars_buffer, accumlator)

for i in range(1, p):
	left_neighbour_id = (id - 1) % p;
	right_neighbour_id = (id + 1) % p;

	stars_buffer = comm.sendrecv(stars_buffer, dest=right_neighbour_id, source=left_neighbour_id)

	#comm.isend(stars_buffer, dest = right_neighbour_id)
	#stars_buffer = comm.recv(source = left_neighbour_id)

	compute_interaction_between_stars(own_stars, stars_buffer, accumlator)

print_result(id, own_stars, accumlator)