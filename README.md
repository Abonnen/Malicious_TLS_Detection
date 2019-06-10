# Overview
The project aims to detect the malicious TLS traffic using machine learing.

# Instruction
## 0. prerequisite
The following libraries and tools are required to be installed:

+ [zeek](https://github.com/zeek/zeek)
+ [scikit-learn](https://scikit-learn.org/stable/)
+ [LightGBM](https://github.com/microsoft/LightGBM)

## 1. get the dataset
The dataset we used comes from [CTU Dataset](https://mcfp.felk.cvut.cz/publicDatasets/) and [Malicious traffic analysis](https://www.malware-traffic-analysis.net/training-exercises.html)

## 2. analyse pcap file
Use Zeek(Bro) to analyse the downloaded pcap file. Zeek will generate some log files. We will use some of them for experiment.

`bro -r filename.pcap`

The entire dataset directory should look like this:
+ Dataset
    + **Malicious**
        + dataset0
            + dataset0.pcap (optional)
            + *.binetflow / **IPadr.txt** (used to make labels)
            + **bro**
                + **conn.log**
                + **dns.log**
                + **ssl.log**
                + **x509.log**
        + dataset1
            + ...
        + ...
    + **Normal**
        + dataset ...

**NOTE**: Do not change filename or foldername in bold type.

## 3. make labels and extract features
+ config dataset path
    open `./feature_extract/config.cfg`, change the path to your dataset directory
+ make labels
    run the command `python ./feature_extract/__label__.py`, then you will get conn_label.log for each conn.log

   **NOTE**: In this project, the infected or normal host IP address has been already known.
+ extract features
    run the command `python ./feature_extract/__main__.py`
    then you will get `./data_model` folder contains the training sample

## 4. create machine learning classifier
In this project, we use RandomForest and GBDT algorithm to develop machine learning classifier
+ random forest
    run the command `python ./machine_learning/random_forest/random_forest.py`, then you will get a `RandomForestClassifier.joblib` model

    run the command `python ./machine_learning/random_forest/test_predict.py`, then you will get the prediction results

+ GBDT
    run the command `python ./machine_learning/LightGBM/gbdt.py`, then you will get a `LGBMClassifier.joblib` model

    run the command `python ./machine_learning/LightGBM/test_predict.py`, then you will get the prediction results

**NOTE**:
+ In `./machine_learning/include`, we present the trained model used for our experiment.
+ If you want to redo the feature seletion and parameter pruning, please use `feature_seletion.py` and `grid_search.py` in each machine learning method folder.