import random
from star import Star

class StarGenerator:
	def __init__(self, min_weight, max_weight, max_coordinate):
		self.min_weight = min_weight
		self.max_weight = max_weight
		self.max_coordinate = max_coordinate

	def generate_concrete_list_of_stars(self):
		stars = []
		for id in range(4):
			for i in range(3):
				stars.append(Star(id * 10 + i, id * 10 + i, id * 10 + i, id * 10 + i))

		return stars

	def generate_concrete_list_of_stars_for(self, id):
		stars = []
		for i in range(3):
			stars.append(Star(id * 10 + i, id * 10 + i, id * 10 + i, id * 10 + i))

		return stars

	def generate_random_list_of_stars(self, number_of_stars):
		return [ self.generate_random_star() for star in range(number_of_stars)]

	def generate_random_star(self):
		x = random.uniform(0, self.max_coordinate)
		y = random.uniform(0, self.max_coordinate)
		z = random.uniform(0, self.max_coordinate)
		weight = random.uniform(self.min_weight, self.max_weight)

		return Star(x, y, z, weight)