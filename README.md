# topic-modeler
Framework to apply LDA and Biterm topic modelling to an unlabeled corpus.

The code for LDA utilized the implementation offered by Gensim [here](https://radimrehurek.com/gensim/models/ldamodel.html) and the code for the Biterm topic model uses the implementation available [here](https://github.com/markoarnauto/biterm).

The folder is organized as follows:
- `requirements.txt`: python packages needed for this project. Install using 
```
pip install -r requirements.txt
```
- `/models/`: Separated by biterm and LDA, includes methods to retrieve top vocabulary words and coherence scores
- `/preprocessing/`: Handles text preprocessing
- `/util/`: Extra utility methods

Scripts in main directory:
- `run_model.py`: Sample code to train LDA/Biterm/Guided LDA models
- `get_coherence.py`: Retrieves coherence metrics for LDA and Biterm models. Topic coherence models from implementation offered by Gensim [here](https://radimrehurek.com/gensim/models/coherencemodel.html).