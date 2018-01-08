PYTHON=/usr/bin/python27
PYTHON=python

EPOCHS=100
NAME=experiment2

LOGFOLDER=log
DATASET=mnist
MODEL=usbn
ALG=adam

LR=3e-4
B1=0.9
B2=0.999
SUPERBATCH=1024
NB=128

# ----------------------------------------------------------------------------

train:
	$(PYTHON) run.py train \
	  --dataset $(DATASET) \
	  --model $(MODEL) \
	  -e $(EPOCHS) \
	  --logname $(LOGFOLDER)/$(DATASET).$(MODEL).$(ALG).$(LR).$(NB).$(NAME) \
	  --plotname $(LOGFOLDER)/$(DATASET)_$(MODEL)_$(ALG)_$(LR)_$(NB)_$(NAME).png \
	  --alg $(ALG) \
	  --lr $(LR) \
	  --b1 $(B1) \
	  --b2 $(B2) \
	  --n_superbatch $(SUPERBATCH) \
	  --n_batch $(NB)
