# Estimator train

## data from numpy
    your row is a map:list format.
    e.g{"input":[1,2,3,4,5]}

## data from pandas
    your columns is a map:list format.
    e.g{"a":1,"b":2,"c":3,"d":4}
    a DataFrame:
        a   b   c   d
     0  1   2   3   4
     1  1   2   3   4


## a service usually use pb(protobuf) data
   you can use `covertToPB.py` covert your checkpoint static to
   pb,then use tf.estimator to load and predict.
   if you don't want to use python as the language for your project
   ,try use golang:tfgo.the github link is:
   [https://github.com/galeone/tfgo](https://github.com/galeone/tfgo)