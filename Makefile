run:
	micromamba run -n comet python -m comet.cli run \
	  --ordering both \
	  --prereg config/prereg.yaml \
	  --paths config/paths.example.yaml \
	  --out artifacts/summary.json
