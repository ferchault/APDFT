#!/usr/bin/env python
""" Posts CI runs on github. Requires a single log file as argument. """

import os
import sys
import subprocess

from github import Github, InputFileContent

def connect(online):
	if online:
		token = os.getenv('GH_TOKEN')
		g = Github(token)
		reponame = 'ferchault/apdft'
		repo = g.get_repo(reponame)
	else:
		g = None
		repo = None
	sha = sys.argv[1]
	return g, repo, sha

def post_results(g, repo, results, logfiles):
	if g is not None:
		# Contact github
		user = g.get_user()
		payload = {'_RESULTS_': InputFileContent(results)}
		for fn, contents in logfiles.items():
			payload[fn] = InputFileContent(contents)

		gist = user.create_gist(False, payload)

		return gist.html_url
	else:
		print (results)
		for fn, contents in logfiles.items():
			print ('#######################', fn)
			print (contents)
		return ''

def create_issue(repo, url):
	if repo is not None:
		body = "Log to be found here: %s." % url
		repo.create_issue(title="Local CI build failed", body=body)

def post_status(repo, sha, state, url):
	if repo is not None:
		repo.get_commit(sha=sha).create_status(state=state, target_url=url, description='Integration tests.', context='continuous-integration/alchemy')

def do_run(srcbasedir, method, code):
	basedir = 'run-%s-%s' % (method, code)
	os.mkdir(basedir)
	with open('%s/n2.xyz' % basedir, 'w') as fh:
		fh.write('2\n\nN 0. 0. 0.\nN 1. 0. 0.')
	cmds = []
	cmds.append('python %s/apdft.py energies --energy_code=%s --apdft_method=%s --energy_geometry=n2.xyz' % (srcbasedir, code, method))
	cmds.append('bash commands.sh')
	cmds.append('python %s/apdft.py energies --energy_code=%s --apdft_method=%s --energy_geometry=n2.xyz' % (srcbasedir, code, method))
	stdout, stderr = '', ''
	for cmd in cmds:
		cp = subprocess.run(cmd.split(), capture_output=True, cwd=basedir)
		stdout += '\n' + cp.stdout.decode('utf-8')
		stderr += '\n' + cp.stderr.decode('utf-8')
		if cp.returncode != 0:
			return cp.returncode == 0, '\n'.join((stderr, stdout))
	return cp.returncode == 0, '\n'.join((stderr, stdout))

if __name__ == '__main__':
	online = (sys.argv[3] == 'ONLINE')
	g, repo, sha = connect(online)
	basedir = sys.argv[2]

	# build matrix
	results = []
	logfiles = {}
	for method in 'HF CCSD'.split():
		for code in 'G09 MRCC'.split():
			spec = (method, code)
			ok, logfilecontents = do_run(basedir, *spec)
			status = ['OK ', 'ERR'][not ok] 
			results.append('%02d: %s %s' % (len(results)+1, status, '/'.join(spec)))
			if not ok:
				logfiles['%02d-%s.txt' % (len(results), '-'.join(spec))] = logfilecontents
	
	# submit
	url = post_results(g, repo, '\n'.join(results), logfiles)
	status = 'error success'.split()[len(logfiles) == 0]
	post_status(repo, sha, status, url)
	if status != 'success':
		create_issue(repo, url)
	
