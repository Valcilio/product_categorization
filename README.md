# Objetive

This project had how objective to create a classification model to give categories for products of any kind.

For this was necessary to create the model with a process of Data Science to predict the categories and create an API with a process of MLOps to productize the model and possibility to do requests in real time for it.

# Architecture

This project's architecture is a union of the architecture from the "Cookiecutter" and the "Clean Architecture", creating, then, a perfect architecture for model development and deployment together, below is possible to see the locations of every thing:

- **data**: all data for modeling (isn't in github). 
- **docs**: all documents of this project, have documentations of model's creation, API's creation and commits.
-  **domain**: all code of the API.
- **logs**: all logs of the modeling process.
- **models**: models developed by the data science process saved in picklefile.
- **notebooks**: notebook necessary to test the models.
- **reports**: all images of the modeling process saved (isn't in github).
- **scalers**: all scalers created in modeling process to rescale the data get sent in the API in a request.
- **cloudbuild.yaml**: auxiliary file for the CI / CD process.
- **Dockerfile**: contais all instrunctions to build the docker image.
- **requirements.txt**: contains all libs necessary for the API's development.
- **.github**: contains all instrunctions to CI / CD process with Github Actions.