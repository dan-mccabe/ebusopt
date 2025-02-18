# `ebusopt` repository
This repository contains code used for Dan McCabe's dissertation research on battery-electric buses (BEBs) at the University of Washington. The major components include GTFS file processing code, optimization methods to determine locations for and schedule usage of opportunity chargers in a transit network, and an app called ZEBRA that processes GTFS data to present summary info and visualizations about the potential to electrify the underlying transit system.

To run most of the optimization methods, your machine must have a valid Gurobi Optimizer license installed. For full functionality, you also need your own API keys for services like Openrouteservice, Google Maps, and Mapbox, which are all free up to certain usage limits.

It is expected that with the current state of the repo, you may run into issues with file structures and missing data. Please submit a GitHub issue for anything you can't easily resolve. I'm happy to do more repo cleanup and find better way sto share larger GTFS files, but only as long as there's a need.

This repository is open-source under the BSD 3-Clause license.

## References
If you've found this code useful for your work, please cite the relevant publication for what you used:

**Charger location optimization** 
```
@article{mccabe2023optimal,
  title={Optimal locations and sizes of layover charging stations for electric buses},
  author={McCabe, Dan and Ban, Xuegang Jeff},
  journal={Transportation Research Part C: Emerging Technologies},
  volume={152},
  pages={104157},
  year={2023},
  publisher={Elsevier}
}
```

**Charging scheduling optimization** 
```
@article{mccabe2024minimum,
  title={Minimum-Delay Opportunity Charging Scheduling for Electric Buses},
  author={McCabe, Dan and Kulcsar, Balazs and Ban, Xuegang Jeff},
  journal={arXiv preprint arXiv:2403.17527},
  year={2024}
}
```

**ZEBRA app or GTFS data processing code:** 
```
@phdthesis{mccabe2024computational,
  title={Computational Tools for Battery-Electric Bus Systems: From Infrastructure Planning to Daily Operations},
  author={McCabe, Dan},
  year={2024},
  school={University of Washington}
}
```


## Repository Structure
* `ebusopt`
  * Home to all source code including optimization models, data handling, and ZEBRA app.
  * `gtfs_beb`
    * Package for handling GTFS data and processing it into formats expected by both ZEBRA app and optimization code.
  * `opt`
    * Python modules related to optimization models and related functions.
  * `scripts`
    * Scripts that run source code from the other directories, e.g. for journal paper case studies.
  * `vis`
    * Visualization functions used for various purposes, e.g., plotting BEB trips.
  * `zebra`
    * Source code of ZEBRA app. The version of ZEBRA in this repository is in development status with some additional features beyond what's included in the production app, and all features might not work as expected.

## Setup
Some configuration is needed prior to running the code in this repository. In particular, some of the code relies on public APIs that require a key for usage. Because these APIs have limited usage allowed before they start incurring costs, users of this repo need to provide their own API keys. These API keys should be stored in a `.env` file in the root directory of the repo (i.e., `ebusopt/.env`). We use the `python-dotenv` package to read these in as environment variables where needed.

The `.env` file should be formatted as follows with the following entries:

```
# Openrouteservice
ORS_KEY=your_ors_key
# Google Maps Directions API
GMAPS_KEY=your_googlemaps_key
# Mapbox (used via Plotly)
MAPBOX_KEY=your_mapbox_key
```

To obtain these keys, follow the steps for each provider. See https://openrouteservice.org/dev/#/signup for Openrouteservice, https://developers.google.com/maps/documentation/directions/overview for Google Maps Directions, and https://docs.mapbox.com/help/getting-started/access-tokens/ for Mapbox.

## Building and running the ZEBRA app
### Option 1: use local environment
First, make sure your Python environment is configured properly. Ensure you are running a recent version of Python (this package is intended to be used with Python 3.11) and create a new virtual environment:

`python3 -m venv /path/to/env`

where `/path/to/env` is the location where your virtual environment will be saved, e.g. `venv`. Then, activate your virtual environment:

`source /path/to/env/bin/activate`

Install all dependencies:

`pip3 install -r requirements.txt`

Launch the app:

`streamlit run ebusopt/zebra/🏠_Home.py`

The app will run on localhost (default port 8501) and you can open it with your web browser at `localhost:8501`.

### Option 2: use Docker
The `Dockerfile` in the home directory contains the instructions needed to build a Docker container that runs the ZEBRA app. This will handle all dependencies for you so that configuring a Python environment is not necessary. First, make sure you have Docker installed on your machine and the Docker daemon is running. Ensure you are in the root directory of this repo. Build the docker image using:

`docker build -t zebra .`

It may take a few minutes to install all dependencies and build the Docker image. Once the process is complete, you can run the app by creating a Docker container based on this image. Run the app with

`docker run -p 8080:8080 zebra`

The app will then be accessible at `localhost:8080`.

**NOTE:** On my computer which has an Apple M3 chip, I need to change the platform specification in the first line of the Dockerfile in order to be able to run the container locally. When running the container on a computer that uses Apple Silicon, specify `--platform=linux/arm64`. For deploying to the server and testing on other hardware, generally use `--platform=linux/amd64`.


## Deploying the ZEBRA app to a server
There are various ways to build and deploy ZEBRA on a web server. This is the process I use when deploying to our UW hosted server.

### 1) Build and test the image locally
Follow the steps in the above section to test the app on your local machine.

### 2) Push the image to Docker Hub
Find the ID of the image you just built and tested using `docker image ls`. Then tag the image with

`docker tag <imageid> <yourusername>/zebra:latest`

Then push it to Docker Hub with
`docker push <yourusername>/zebra:latest`

Clean up any local containers/images to save disk space using `docker rm <container_id>`/`docker rmi <image_id>`. You can see a list of these and their IDs with `docker container ls -a` `and docker image ls -a`

### 3) Pull the image to the server
Connect to the server that will host the app using `ssh`. Then pull the image from Docker Hub with

`sudo docker pull <yourusername>/zebra:latest`

### 4) Run the container on the server
`sudo docker run -d -p 80:8080 --restart on-failure -m 3g <yourusername>/zebra:latest`
The options above are what I generally use to create the app. `-d` runs the app in the background, `p` handles port mapping, `--restart on-failure` ensures the app restarts if it ever crashes, and `-m 3g` limits RAM usage of the app to 3 GB (our server has only 4 GB of RAM).

