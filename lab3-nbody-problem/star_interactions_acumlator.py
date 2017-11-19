from scipy import constants



class StarInteractionsAccumlator:
	
	def __init__(self, acucmulated_sum_x = 0.0, acucmulated_sum_y = 0.0, acucmulated_sum_z = 0.0):
		self.acucmulated_sum_x = acucmulated_sum_x
		self.acucmulated_sum_y = acucmulated_sum_y
		self.acucmulated_sum_z = acucmulated_sum_z

	@staticmethod
	def squared_euclidean_distance(star_1, star_2):
		return (star_2.x - star_1.x) ** 2 + (star_2.y - star_1.y) ** 2 + (star_2.z - star_1.z) ** 2

	@staticmethod
	def sum_of_interaction(star_1, star_2):
		euclidean_distance = StarInteractionsAccumlator.squared_euclidean_distance(star_1, star_2) + 1
		
		acucmulated_sum_x = (star_2.x - star_1.x) / euclidean_distance
		acucmulated_sum_y = (star_2.y - star_1.y) / euclidean_distance
		acucmulated_sum_z = (star_2.z - star_1.z) / euclidean_distance

		return (
			acucmulated_sum_x,
			acucmulated_sum_y,
			acucmulated_sum_z)

	def acceleration_x(self, planet_mass):
		return constants.G * planet_mass * self.acucmulated_sum_x

	def acceleration_y(self, planet_mass):
		return constants.G * planet_mass * self.acucmulated_sum_y

	def acceleration_z(self, planet_mass):
		return constants.G * planet_mass * self.acucmulated_sum_z

	def update(self, star_1, star_2):
		euclidean_distance = self.squared_euclidean_distance(star_1, star_2) + 1
		
		self.acucmulated_sum_x += star_2.weight * (star_2.x - star_1.x) / euclidean_distance
		self.acucmulated_sum_y += star_2.weight * (star_2.y - star_1.y) / euclidean_distance
		self.acucmulated_sum_z += star_2.weight * (star_2.z - star_1.z) / euclidean_distance

	def sum(self, another_accumlator):
		return StarInteractionsAccumlator(
			self.acucmulated_sum_x + another_accumlator.acucmulated_sum_x,
			self.acucmulated_sum_y + another_accumlator.acucmulated_sum_y,
			self.acucmulated_sum_z + another_accumlator.acucmulated_sum_z)

	def __repr__(self):
		return str(self.__dict__)
