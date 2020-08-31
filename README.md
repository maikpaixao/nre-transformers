# Neural Relation Extraction
## Laboratoire d’Informatique et Systèmes - Federal Rural University of Pernambuco


## Setup 

Clone the repository from our github page (don't forget to star us!)

```bash
git clone https://gitlab.lis-lab.fr/maik.paixao/lisnre.git
```

Then install all the requirements:

```
pip install -r requirements.txt
```

Note that we have excluded all data and pretrain files for fast deployment. You can manually download them by running scripts in the ``benchmark`` folders. To download Semeval dataset, you can run

```
bash benchmark/download_semeval.sh
```

## Dataset

All instances in the dataset are divided into 3 files (train, test, val) in .json format. In order to generate these datasets you need to run the scripts in the ``knowledge`` folder.

```
python execute.py --prefix "train or test or val"
```

## Execution

To execute the training scripts firstly you need to change the actual directory to the ``scripts`` folder.

```
cd scripts/
```
After this it is possible to execute the file by typing the following command. 
```
python bert.py
```
It is possible to specify the type of features to be used (position, path, chunks, semantics) by typing the name of the method as argument in the command.
```
python bert.py --position
```
## RNN's Models

RNN-based models were execute using the platform Google Collab, so the code is formatted in Jupyter Notebooks and can be easily executed.
The notebooks can be found at the ``notebooks`` folder.

```
cd notebooks/
```

