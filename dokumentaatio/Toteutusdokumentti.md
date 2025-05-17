# Toteutusdokumentti

## Ohjelman yleisrakenne

Ohjelma perustuu neuroverkkoon, joka oppii tunnistamaan käsinkirjoitettuja lukuja. Koulutus tapahtuu MNIST-datasetillä, joka sisältää 60 000 esimerkkiä neuroverkon kouluttamiseen ja 10 000 esimerkkiä testaukseen. Mallia voi jatkokouluttaa sekä manuaalisesti testata käyttöliittymällä.

Käytän neuroverkkoa, jossa on kolme kerrosta. 784 neuronia sisältävä sisääntulokerros, 128 neuronia sisältävä piilokerros ja 10 neuronia sisältävä ulostulokerros.

Mallin oppiminen perustuu gradienttimenetelmää virhefunktion minimoimiseen painojen ja biasien säätelyllä. backpropagation -algoritmiin

## Puutteet ja parannusehdotukset

## Laajojen kielimallien käyttö

ChatGPT:tä on käytetty joidenkin käsitteiden/aiheiden selittämiseen yksityiskohtaisesti.

## Lähteet

- [Michael Nielsen, Using neural nets to recognize handwritten digits](http://neuralnetworksanddeeplearning.com/chap1.html)
- [Matt Hodges, Building a Neural Network From Scratch with Numpy](https://matthodges.com/posts/2022-08-06-neural-network-from-scratch-python-numpy/#acknowledgements)
- [Numpy tutorials](https://github.com/numpy/numpy-tutorials/tree/main)
- [Deep learning -kirja](https://www.deeplearningbook.org/)
