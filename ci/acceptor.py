#!/usr/bin/env python

def accept(request):
	import os
	import hmac
	import hashlib

	# Check validity
	hooksecret = os.getenv('GH_HOOKSECRET')
	signature = request.headers.get("X-Hub-Signature")
	if not signature or not signature.startswith("sha1="):
		return 'err'

	digest = hmac.new(hooksecret.encode(), request.data, hashlib.sha1).hexdigest()

	if not hmac.compare_digest(signature, "sha1=" + digest):
		return 'err2'

	# Contact github
	from github import Github
	token = os.getenv('GH_TOKEN')
	g = Github(token)
	data = request.get_json()
	repo = g.get_repo("ferchault/multiQM")
	sha = data["pull_request"]["head"]["sha"]
	repo.get_commit(sha=sha).create_status(
		state="pending",
		target_url="https://chemspacelab.org",
		description="local CI is building",
		context="ci/localCI"
	)

