#!/usr/bin/env python
import argparse
import sys
import apdft
import apdft.commandline as acmd
import apdft.settings as aconf

if __name__ == '__main__':
	# load configuration
	parser = acmd.build_main_commandline()
	conf = aconf.Configuration()
	conf.from_file()
	mode, modeshort, conf = acmd.parse_into(parser, configuration=conf)

	# execute
	if mode == 'energies':
		acmd.mode_energies(conf, modeshort)
	else:
		apdft.log.log('Unknown mode %s' % mode, level='error')
	
	# persist configuration
	conf.to_file()