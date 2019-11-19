#!/usr/bin/env python
import sys
import cProfile
import pstats

if __name__ == "__main__":
    pr = cProfile.Profile()
    pr.enable()
    import apdft.commandline as acmd
    ret = acmd.entry_cli()
    pr.disable()
    stats = pstats.Stats(pr)
    stats.sort_stats('cumulative')
    stats.print_stats(30)
    sys.exit(ret)
