import math
import sys

from mpi4py import MPI
from star import Star
from star_generator import StarGenerator
from star_interactions_acumlator import StarInteractionsAccumlator


def number_of_process_additional_star(number_of_stars, number_of_processes, process_id):
	if process_id < number_of_stars % number_of_processes:
		return 1
	else:
		return 0

def find_number_of_stars_for_process(number_of_stars, number_of_processes, process_id):
	main_part = number_of_stars / number_of_processes
	main_part += number_of_process_additional_star(number_of_stars, number_of_processes, process_id)

	return main_part

def compute_interaction_between_stars(own_stars, stars_buffor, own_accumlators, accumlators_buffer = None):
	for i in range(len(own_stars)):
		for j in range(len(stars_buffor)):
			#print('Debug:',id, i , j, str(own_stars[i]), str(stars_buffor[j]))
			sum_of_interaction = StarInteractionsAccumlator.sum_of_interaction(own_stars[i], stars_buffor[j])
			own_accumlators[i].acucmulated_sum_x += sum_of_interaction[0] * stars_buffor[j].weight
			own_accumlators[i].acucmulated_sum_y += sum_of_interaction[1] * stars_buffor[j].weight
			own_accumlators[i].acucmulated_sum_z += sum_of_interaction[2] * stars_buffor[j].weight

			if accumlators_buffer:
				accumlators_buffer[j].acucmulated_sum_x -= sum_of_interaction[0] * own_stars[i].weight
				accumlators_buffer[j].acucmulated_sum_y -= sum_of_interaction[1] * own_stars[i].weight
				accumlators_buffer[j].acucmulated_sum_z -= sum_of_interaction[2] * own_stars[i].weight

def print_result(id, own_stars, own_accumlators):
	for i in range(len(own_stars)):
		print(id, str(own_stars[i]), str(own_accumlators[i]))

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
own_accumlators = [StarInteractionsAccumlator() for acc in range(find_number_of_stars_for_process(NUMBER_OF_STARS, p, id))]
stars_buffer = own_stars
accumlators_buffer = [StarInteractionsAccumlator() for acc in range(find_number_of_stars_for_process(NUMBER_OF_STARS, p, id))]

compute_interaction_between_stars(own_stars, stars_buffer, accumlators_buffer)

for i in range(int(math.floor(p/2.0))):
	left_neighbour_id = (id - 1) % p;
	right_neighbour_id = (id + 1) % p;

	stars_buffer = comm.sendrecv(stars_buffer, dest=right_neighbour_id, source=left_neighbour_id)
	accumlators_buffer = comm.sendrecv(accumlators_buffer, dest=right_neighbour_id, source=left_neighbour_id)
	#comm.isend(stars_buffor, dest = right_neighbour_id, tag=0)
	#comm.isend(accumlators_buffer, dest = right_neighbour_id, tag=1)
	#stars_buffor = comm.recv(source = left_neighbour_id, tag=0)
	#accumlators_buffer = comm.recv(source = left_neighbour_id, tag=1)
	#buffor = comm.sendrecv(buffor, right_neighbour_id, source=left_neighbour_id)

	if i != int(math.floor(p/2.0)) - 1:
		compute_interaction_between_stars(own_stars, stars_buffer, own_accumlators, accumlators_buffer)
	else:
		compute_interaction_between_stars(own_stars, stars_buffer, own_accumlators)

accumlators_buffer = comm.sendrecv(accumlators_buffer, dest = int(id - math.floor(p/2.0)) % p, source=int(id + math.floor(p/2.0)) % p)

for i in range(len(own_accumlators)):
	own_accumlators[i] = own_accumlators[i].sum(accumlators_buffer[i])

print_result(id, own_stars, own_accumlators)
