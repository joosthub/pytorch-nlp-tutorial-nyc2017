# pytorch-nlp-tutorial

## Data
```
1. mkdir day_1/data
2. copy contents of this drive in the data folder day_1/data

https://drive.google.com/open?id=0B1sSP-aCtfuHRFJWTkdUbjFUZDQ

3. Download glove and unpack contents in day_1/data/glove
http://nlp.stanford.edu/data/glove.6B.zip
```

```
# install anaconda (if needed)

conda create -n dl4nlp python=3.6
source activate dl4nlp
conda install ipython 
conda install jupyter
python -m ipykernel install --user --name dl4nlp
 
# install pytorch
# visit pytorch.org

# install torch text
git clone https://github.com/pytorch/text.git
cd text
python setup.py install

cd ..
pip install -r requirements.txt
```

