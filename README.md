# pytorch-nlp-tutorial
## Day 1
[Day 1 Slides](https://drive.google.com/open?id=1uCzIGJT2ni9_ZlcwUHEcUJuyJbjHTKc3izsFrPyHw4M)

### Day 1 Data
```
1. mkdir day_1/data
2. copy contents of this drive in the data folder day_1/data

https://drive.google.com/open?id=0B1sSP-aCtfuHRFJWTkdUbjFUZDQ

3. Download glove and unpack contents in day_1/data/glove
http://nlp.stanford.edu/data/glove.6B.zip
```

## Day 2
[Day 2 Slides](https://drive.google.com/open?id=1ZRzwllU7tMlQJevGhLYJo_woEA4ABPXUQHGFb8Ijbws)


### Day 2 Data

1. [Trump Tweets](https://drive.google.com/a/joostware.com/file/d/0B2hg7DTHpfLsNUxFcndiNlVxSmM/view?usp=sharing)
2. Not-pruned Names dataset
    a. [Train](https://drive.google.com/open?id=0B2hg7DTHpfLsTVNfNnpDVGZmZTQ)
    b. [Test](https://drive.google.com/open?id=0B2hg7DTHpfLsWmtQT1lXREx2Qmc)
    c. [Day One Version](https://drive.google.com/open?id=0B2hg7DTHpfLsMzg5QlRyMzhfQ1U)
3. [Stanford NLI dataset](https://drive.google.com/open?id=0B2hg7DTHpfLsTy1BTlk0dTBwREU)
4. [Amazon Reviews small train](https://drive.google.com/open?id=0B2hg7DTHpfLsbk1yME5HN0dxVmc)

```
# install anaconda (if needed)

conda create -n dl4nlp python=3.6
source activate dl4nlp
conda install ipython
conda install jupyter
python -m ipykernel install --user --name dl4nlp

# install pytorch
# visit pytorch.org

# assume we are inside a folder dl4nlp
# note: that if you alternatively download the zip and unzip it to
#   a folder, it will be named something else
git clone https://github.com/joosthub/pytorch-nlp-tutorial.git
cd pytorch-nlp-tutorial

pip install -r requirements.txt

# going back to root folder
cd ..

# install torch text
git clone https://github.com/pytorch/text.git
cd text
python setup.py install
```

