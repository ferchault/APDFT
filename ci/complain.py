#!/usr/bin/env python
""" Posts failed CI runs on github. Requires a single log file as argument. """

import os
import sys

from github import Github, InputFileContent

def post_error(logfile):
	# Contact github
	token = os.getenv('GH_TOKEN')
	g = Github(token)
	user = g.get_user()
	with open(logfile) as fh:
		gist = user.create_gist(True, {'logfile.txt': InputFileContent(fh.read())}, 'Output of the failing CI job.')

	repo = g.get_repo("ferchault/multiQM")
	body = "Log to be found here: %s." % gist.html_url
	repo.create_issue(title="Local CI build failed", body=body)

if __name__ == '__main__':
	logfile = sys.argv[1]
	post_error(logfile)
