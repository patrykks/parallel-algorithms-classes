from star_interactions_acumlator import StarInteractionsAccumlator
from star_velocity import StarVelocity

class Star:
	
	def __init__(self, x, y, z, weight):
		self.x = x
		self.y = y
		self.z = z
		self.weight = weight
		self.last_accumlators = [StarInteractionsAccumlator(), StarInteractionsAccumlator()]
		self.velocity = StarVelocity()

	def update_accumlators(self, accumlator):
		self.last_accumlators[0] = self.last_accumlators[1]
		self.last_accumlators[1] = accumlator

	def __repr__(self):
		return str(self.__dict__)