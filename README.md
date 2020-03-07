# [Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/overview)

## Credits: 
* [Challenge Page on Kaggle](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/overview)
* [NB-SVM strong linear baseline](https://www.kaggle.com/jhoward/nb-svm-strong-linear-baseline) and [Stop the S@#$ - Toxic Comments EDA](https://www.kaggle.com/jagangupta/stop-the-s-toxic-comments-eda) are used to understand the dataset used in challenge. 
* [Dataset](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)

## How to run:
* `python3 -i main.py` : It will print the summary of result and saves it in `log.json` file.
* Note: Running for first time will take 2-4 minutes. Most of this time is taken for preprocessing comments which is currently being done in a python loop which can be hugely optimized. Subsequent runs will take less time since it uses the previously saved preprocessed comments.
* Model based on unique features can be run by pasting the code from `interact.py` in the python shell which will come after running `python3 -i main.py`.

## Requirements:
* sklearn
* nltk
* numpy 
* pandas 
* To install dependencies, run `python3 -m pip install numpy pandas sklearn nltk`