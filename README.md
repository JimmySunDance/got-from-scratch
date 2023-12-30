# GPT from Scratch

This repo follows the tutorial:
https://www.youtube.com/watch?v=UU1WVnMk4E8&t=15696s
by FreeCodeCamp.org.

It utilised PyTorch to build out a GPT chat bot from first principles, using the Transformer attention architecture proposed in the seminal "Attention is all you need" paper by A. Vaswani et. al.

## Pre-requisites
- Python 3.10
- PyTorch

## Set up virtual env and install requirements 
```
$ python3 -m venv .venv
$ source .venv/bin/activate
$ pip3 install -r req.txt
```

## Download the data
This project uses 2 data sources. 
- [Dorothy and the Wizard in Oz](https://www.gutenberg.org/cache/epub/22566/pg22566.txt) - Taken from [Project Gutenberg](https://www.gutenberg.org)

This First source is used in the PyTorch exploration section, specifically the Bigram tutorial. 

- [The OpenWebText Corpus](https://skylion007.github.io/OpenWebTextCorpus/)

This is a *very* large dataset ~12GB downloaded then ~40GB uncompressed. 

In order to extract the data into the required training, validation and vocabulary files, first download and unzip the `.tar` file from the link above. Next create a folder in your working directory `data/openwebtext` and move contents of the unziped openwebtext folder into this new one. Finally, run the following command:
```
$ python data-extract.py
```
The file structure should look like the following:
```
|-data
|  |-openwebtext
|    |-file_1.xz
|    |-file_2.xz
|    |......
|    |-file_XX.xz
|
| extract-data.py
```

Once this has run (should take ~20mins depending on your system) you will have 3 new files in your directory totalling roughly ~46GB in size.
```
|-data
|  |-openwebtext
|    |-output_train.txt
|    |-output_val.txt
|    |-vocab.txt
```
You are now ready to begin training the model. 

## To train the model
Ensure you have first downloaded and extracted the text corpus required for training the model - as described in the previous section. 

To train the model simply run:
```
$ python training.py
```
This will take some time to complete and will produce a `model-01.pkl` file once complete. 

You can continue to improve your model by running this step multiple times. You may also find it worth while to play with and adjust the hyper-parameters to those that best suit your systems capabilities. 

## Run the chatbot
Once your model is trained you can interact with it via `chatbot.py`.

Run this with the command:
```
$ python chatbot.py
```

The terminal will then prompt you to enter the beginning of a string (your prompt), after a few seconds, the program will output the completed string. 

The quality of your output will correspond to the quality of the model you have trained. 

### Exit
In order to exit the chatbot, enter the prompt `quit()`.