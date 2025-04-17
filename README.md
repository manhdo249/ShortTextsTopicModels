# This repo forked from TopMost!

## Preparing Libraries

1. Python 3.10.14
2. Install the following libraries
    ```
    numpy==1.26.3
    scipy==1.10.1
    sentence-transformers==2.7.0
    torchvision==0.19.1
    gensim==4.3.3
    scikit-learn==1.5.1
    tqdm==4.66.5
    wandb==0.18.1
    topmost==0.0.5
    ```
2. Install java
3. Download [this java jar](https://hobbitdata.informatik.uni-leipzig.de/homes/mroeder/palmetto/palmetto-0.1.0-jar-with-dependencies.jar) to `./topmost/evaluations` and rename it to `palmetto.jar`
4. Download and extract [this processed Wikipedia corpus](https://hobbitdata.informatik.uni-leipzig.de/homes/mroeder/palmetto/Wikipedia_bd.zip) to `./topmost/evaluations/wiki_data` as an external reference corpus.

    Here is the folder structure:
    ```
        |- topmost
            |- evaluations
                | - wiki_data
                    | - wikipedia_bd/
                    | - wikipedia_bd.histogram
                |- palmetto.jar
            
    ```

## Running
To run and evaluate our model, run the following command:

```
bash bash/NewMethod/top100/NewMethod_Biomedical_top100_cluster50.sh
```


## Acknowledgement
Some part of this implementation is based on [TopMost](https://github.com/BobXWu/TopMost). We also utilizes [Palmetto](https://github.com/dice-group/Palmetto) for the evaluation of topic coherence.
