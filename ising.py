import numpy as np
from energy import sparse_half_capy
class ising:

	# initialize various parameters related to the simulation
	def __init__(self, repvotes, demvotes, A, temp=2**20, target_energy=1.0, steps=100):
		# vote totals
		self.repvotes = repvotes
		self.demvotes = demvotes

		# temperature
		self.temp = temp

		# adjacency matrix
		self.A = A

		# desired energy level - between 0 and 1
		self.target_energy = target_energy

		# how long the simulation should run
		self.steps = steps
		
	# setup an iterator where each step will correspond to a step in the simulation
	def __iter__(self):
		self.counter = 0
		self.newrepvotes = np.array(self.repvotes)
		self.newdemvotes = np.array(self.demvotes)
		return self

	# how the next step in the simulation is determined
	def __next__(self):
		'''
		iterator to give the next state in the ising_simulation
		Simple metropolis hastings algorithm that always accepts steps that move in the direction of the target energy
		and accepts steps that move away from the target energy with probability based on a "temperature" parameter, and the difference in energies of the two configurations
		'''

		# stop simulation if reached number of desired steps
		if self.counter == self.steps:
			raise StopIteration

		# otherwise increment step counter
		self.counter += 1
		
		# get new proposed configurations of votes
		prop_rep, prop_dem = self.proposal(self.newrepvotes, self.newdemvotes)

		# compute the old and new energies
		old_energy = sparse_half_capy(self.newrepvotes, self.newdemvotes, self.A)
		new_energy = sparse_half_capy(prop_rep, prop_dem, self.A)
		energy = old_energy

		# determine whether to accept or reject the proposed step
		# how we determine if we accept depends on if we are above or below the target energy
		if (old_energy < self.target_energy):
			swap = self.accept(old_energy, new_energy)
		else:
			swap = self.accept(new_energy, old_energy)

		# if we swap, assign the new data for the simulation state
		if swap:
			energy = new_energy
			self.newrepvotes = prop_rep
			self.newdemvotes = prop_dem
		
		# return current simulation state
		return self.newrepvotes, self.newdemvotes, energy

	# tells how long the simulation is
	def __len__(self):
		return self.steps

	def accept(self, old_energy, new_energy):
		'''
		metropolis hasting acceptance function for higher energies (i.e. always accept higher)
		old - old energy score
		new - new energy score

		### swap old and new to accept lower energies
		'''
		if new_energy >= old_energy:
			return True
		elif np.random.rand() < np.exp(-self.temp*(old_energy-new_energy)):
			return True
		else:
			return False
    
	def proposal(self, curr_rep, curr_dem):
		'''
		proposal function, randomly pick two different nodes and swap some number of republican and democrat voters
		'''

		# choose two nodes uniformly at random
		idx1 = 0
		idx2 = 0
		new_rep = np.array(curr_rep)
		new_dem = np.array(curr_dem)
		while idx1 == idx2:
			idx1 = np.random.randint(0,curr_rep.size-1)
			idx2 = np.random.randint(0,curr_dem.size-1)
		
		# determine the number of people to swap
		# must be less than or equal to the number of people of either party at the two nodes
		max_num_swap = min(curr_rep[idx1], curr_dem[idx2])
		if max_num_swap > 1:
			num_swap = np.random.randint(1, max_num_swap)
		else:
			num_swap = 0

		# swap republicans from node 1 to 2 and swap democrats from node 2 to 1
		new_rep[idx1] -= num_swap
		new_rep[idx2] += num_swap
		new_dem[idx1] += num_swap
		new_dem[idx2] -= num_swap
		return new_rep, new_dem

	# change the target energy for the simulation
	def set_target(self, target):
		self.target = target
	
