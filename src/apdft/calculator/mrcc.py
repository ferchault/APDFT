
class MrccCalculator(Calculator):
	_methods = {
		'CCSD': 'ccsd',
	}

	@staticmethod
	def _parse_densityfile(densityfile):
		""" Returns all relevant data from a MRCC density file.

		Columns 0-2: x, y, z coordinates
		Column 3: weights
		Column 4: density"""
		with open(densityfile, 'r') as fh:
			_ = np.fromfile(fh, 'i4')
			q = _[3:-1].view(np.float64)
			ccdensity = q.reshape((-1, 10))
		return ccdensity[:, 1:6]

	@staticmethod  
	def density_on_grid(densityfile, grid):   
		ccdensity = MrccCalculator._parse_densityfile(densityfile)
		if not np.allclose(grid, ccdensity[:, :3]):
			raise ValueError('Unable to combine different grids.')
		return ccdensity[:, 4]
	
	@staticmethod
	def get_grid(nuclear_numbers, coordinates, outputfolder):
		""" Obtains the integration grid from one of the MRCC output files. """
		ccdensity = MrccCalculator._parse_densityfile('%s/DENSITY' % outputfolder)
		return ccdensity[:, :3], ccdensity[:, 3]

	@staticmethod
	def _format_charges(coordinates, nuclear_numbers, nuclear_charges):
		ret = []
		for coord, Z_ref, Z_tar in zip(coordinates, nuclear_numbers, nuclear_charges):
			ret.append('%f %f %f %f' % (coord[0], coord[1], coord[2], (Z_tar - Z_ref)))
		return '\n'.join(ret)

	def get_input(self, coordinates, nuclear_numbers, nuclear_charges, grid, iscomparison=False):
		basedir = os.path.dirname(os.path.abspath(__file__))
		with open('%s/templates/mrcc.txt' % basedir) as fh:
			template = j.Template(fh.read())
		
		env_coord = GaussianCalculator._format_coordinates(nuclear_numbers, coordinates)
		env_basis = self._basisset
		env_numatoms = len(nuclear_numbers)
		env_charged = MrccCalculator._format_charges(coordinates, nuclear_numbers, nuclear_charges)
		
		return template.render(coordinates=env_coord, method=self._methods[self._method], basisset=env_basis, numatoms=env_numatoms, charges=env_charged)

	@classmethod
	def get_runfile(self, coordinates, nuclear_numbers, nuclear_charges, grid):
		basedir = os.path.dirname(os.path.abspath(__file__))
		with open('%s/templates/mrcc-run.sh' % basedir) as fh:
			template = j.Template(fh.read())
		return template.render()

	def get_density_on_grid(self, folder, grid):
		return MrccCalculator.density_on_grid(folder + '/DENSITY', grid)

	@staticmethod
	def get_total_energy(folder):
		""" Returns the total energy in Hartree."""
		logfile = '%s/run.log' % folder
		try:
			energy_cc = cclib.io.ccread(logfile)
		except:
			apdft.log.log('Unable to read energy from log file.', filename=logfile, level='error')
			return 0
		return energy_cc # / 27.211386245988
	@staticmethod
	def parse_energy_cc_Mrcc(log_file):
		"""Parse the couple cluster energy from an MRCC output file"""
		with open(log_file,'r') as logf:
			while True:
				line=logf.readline()
				if "Final results:" in line:
					good_line=logf.readline()
					if "Total CCSD energy" in good_line:
						for x in good_line.split(' '):
							try:
								float(x)
								return (float(x))
							except:
								pass  
	@staticmethod
	def get_electronic_dipole(folder, gridcoords, gridweights):
		raise NotImplementedError()
