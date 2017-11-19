class StarVelocity:
	
	def __init__(self, v_x = 0, v_y = 0, v_z = 0):
		self.v_x = v_x
		self.v_y = v_y
		self.v_z = v_z

	def __repr__(self):
		return str(self.__dict__)
