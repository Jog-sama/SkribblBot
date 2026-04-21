.PHONY: setup download features train app clean

# Run full pipeline (download → features → train)
setup:
	python setup.py

# Individual steps
download:
	python scripts/make_dataset.py

features:
	python scripts/build_features.py

train:
	python scripts/model.py

# Run the app locally
app:
	python app.py

# Remove all generated data and model files
clean:
	rm -rf data/raw/*.npy data/processed/*.npy data/outputs/* models/*.pkl models/*.pth

# Install dependencies
install:
	pip install -r requirements.txt
