from star import Star
from star_interactions_acumlator import StarInteractionsAccumlator
from star_generator import StarGenerator

MAX_WEIGHT = 100000000000
MIN_WEIGHT = 100000
MAX_COORDINATE = 10000
NUMBER_OF_STARS = 12


star_generator = StarGenerator(MIN_WEIGHT, MAX_WEIGHT, MAX_COORDINATE)
stars = star_generator.generate_concrete_list_of_stars()
stars_acceleration = [StarInteractionsAccumlator() for star_acceleration in range(NUMBER_OF_STARS)]

for i in range(NUMBER_OF_STARS):
	for j in range(NUMBER_OF_STARS):
		stars_acceleration[i].update(stars[i], stars[j])

for i in range(NUMBER_OF_STARS):
	print(str(stars[i]), str(stars_acceleration[i]))