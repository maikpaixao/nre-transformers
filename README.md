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

## Execute

All instances in the dataset are divided into 3 files (train, test, val) in .json format. In order to generate these datasets you need to run the scripts in the ``knowledge`` folder.


```
cd scripts/
```
All instances in the dataset are divided into 3 files (train, test, val) in .json format. 
```
python bert.py
```
All instances in the dataset are divided into 3 files (train, test, val) in .json format. 
```
python bert.py --position
```


