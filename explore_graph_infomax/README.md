# Exploratory study on graph infomax
With the purpose of understanding how the different layers in graph infomax affect performance,choices of encoder function and summary function are examined.
A DGI structure coding involves the encoder, sumary and corruption layers. Encoder and summary layers are modified.

# Environment

`pyton =3.8.8`

`pytorch=1.70 with cuda10.1`

# Encoder layers

`GCNII: dgi_gcn2.py`

`APPNP: dgi_appnp.py`

`GIN: dgi_gin.py`

# Summary Layers

This is done in dgi_readout.py, where need to manually modify the code of summary=lambda z in script.
The summary layer is function of sigmoid(pooled features), where the pooled features comes from
1. mean pooling

2. max pooling

3. min pooling

4. concatentation of [mean, max, min] pooling, with a linear MLP to a 512 hidden size.

# To Run

#To run the encoder layer

E.g: to run the dgi_appnp.py

`python dgi_appnp.py`

*Parameter tuning will run in the main. If already know optimized parameters, need to comment the corresponding parts of the code in __main__. Different dataset also required manually modification to change "dataset" to "Cora", "Citeseer" or "Pubmed".

#To run the summary layer

`python dgi_readout`

*need to manually modify the callable "summary" function in "DeepGraphInfomax" to change the pooling methods. Also need to change the "dataset" for different datasets
