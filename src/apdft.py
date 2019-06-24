#!/usr/bin/env python
import argparse
import sys
import apdft
import apdft.commandline as acmd

if __name__ == '__main__':
	parser = acmd.build_main_commandline()
	mode, modeshort, conf = acmd.parse_into(parser)

	if mode == 'energies':
		acmd.mode_energies(conf, modeshort)
	else:
		apdft.log.log('Unknown mode %s' % mode, level='error')