# NOTE: this file assumes the root is the current directory

# NOTE 2: when running on uci servers, do
#             module load python/3.6.4
#         instead of the default
#             module load python/3.5.1
#         in the .bash_profile

# NOTE 3: to start a screen, type
#             screen -S SCREEN_NAME
#         to detach a screen, type (within the screen)
#             CTRL+A, D
#         to view all screens, type
#             screen -ls
#         to return to a screen, type
#             screen -x SCREEN_NAME
#         to kill a screen, type (within the screen)
#             CTRL+D

import os

def download_amazon_review_data():
    os.system('wget "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Books_5.json.gz" -P db')
    os.system('wget "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz" -P db')
    os.system('wget "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Movies_and_TV_5.json.gz" -P db')
    os.system('wget "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_CDs_and_Vinyl_5.json.gz" -P db')
    os.system('wget "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Clothing_Shoes_and_Jewelry_5.json.gz" -P db')
    os.system('wget "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Home_and_Kitchen_5.json.gz" -P db')
    os.system('wget "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Kindle_Store_5.json.gz" -P db')
    os.system('wget "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Sports_and_Outdoors_5.json.gz" -P db')
    os.system('wget "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Cell_Phones_and_Accessories_5.json.gz" -P db')
    os.system('wget "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Health_and_Personal_Care_5.json.gz" -P db')
    os.system('wget "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Toys_and_Games_5.json.gz" -P db')
    os.system('wget "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Video_Games_5.json.gz" -P db')
    os.system('wget "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Tools_and_Home_Improvement_5.json.gz" -P db')
    os.system('wget "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty_5.json.gz" -P db')
    os.system('wget "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Apps_for_Android_5.json.gz" -P db')
    os.system('wget "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Office_Products_5.json.gz" -P db')
    os.system('wget "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Pet_Supplies_5.json.gz" -P db')
    os.system('wget "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Automotive_5.json.gz" -P db')
    os.system('wget "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Grocery_and_Gourmet_Food_5.json.gz" -P db')
    os.system('wget "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Patio_Lawn_and_Garden_5.json.gz" -P db')
    os.system('wget "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Baby_5.json.gz" -P db')
    os.system('wget "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Digital_Music_5.json.gz" -P db')
    os.system('wget "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Musical_Instruments_5.json.gz" -P db')
    os.system('wget "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Amazon_Instant_Video_5.json.gz" -P db')

def unzip_amazon_review_data():
    os.system('gunzip db/*.gz')

def install_libraries():
    os.system('pip3 install --user nltk')
    # os.system('pip3 install --user numpy') # comes with scipy
    os.system('pip3 install --user scipy')
    os.system('pip3 install --user sklearn')
    # os.system('pip3 install --user matplotlib') # only used in Jupyter notebook
    # os.system('pip3 install --user eli5') # only used in Jupyter notebook

def download_stopwords_corpus():
    import nltk
    nltk.download('stopwords')

if __name__ == '__main__':
    download_amazon_review_data()
    unzip_amazon_review_data()
    install_libraries()
    download_stopwords_corpus()
