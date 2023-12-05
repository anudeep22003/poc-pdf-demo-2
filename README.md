## Tasks

- [ ] Add ability to add a prompt template to help with personalization
- [ ] Improve the recall with better filtering. Look beyong `similarity_top_k` and maybe look at similarity scores?
- [ ] In last step of answer generation skip index formation and straight use the retrieved nodes rather thean constructing an index over it. Doing so makes it redo a similarity match and hence it comes up with fewer sources
- [ ] Implement caching to improve response speed
- [ ] Implement embedding similarity search instead of getting OpenAI to do it for you

# First set up the environment

First set up the environment. You have two options:

- mamba. Install from [here](https://mamba.readthedocs.io/en/latest/installation.html) and run `mamba env create -f env.yml`
  - then run `mamba activate poc-pdf-demo-2`
- if using pip there is a `requirements.txt` file that you can use with `venv`

# To run the server

`uvicorn main:app --port 8000`

- the server will run on port 8000
- it isnt bound to a host so it will be localhost

# api documentation

- documentation is available at [http://localhost:8000/redoc](http://localhost:8000/redoc)
