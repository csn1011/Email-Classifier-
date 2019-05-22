import numpy as np
import pandas as pd

def file_opener(filename):
    """Opens file and prints error message/returns none if the file cannot be opened"""
    try:
        text_file = open(filename)
    except:
        return print("Error, file ", filename, "could not be opened")
    return text_file



def words_in_texts(words, texts):
    '''
    Args:
        words (list-like): words to find
        texts (Series): strings to search in
    
    Returns:
        NumPy array of 0s and 1s with shape (n, p) where n is the
        number of texts and p is the number of words.
    '''
    indicator_list = []
    for i in texts:
        temp = []
        for x in words:
            if x in i:
                temp = temp + [1]
            else:
                temp = temp + [0]
        indicator_list+= [temp]
    indicator_array = np.array(indicator_list)
    return indicator_array

def wordsandre(words, subwords, df):
    daf = df.reset_index(drop = True)
    indicator_list = []
    texts = daf['email']
    size = len(texts)
    subjects = daf['subject']
    for i in range(size):
        z=texts[i]
        s = subjects[i]
        temp = []
        if isinstance(s, float):
            substring,s = [],""
        else:
            substring = list(s)
        upper = 0
        ucase = 0
        textstring = z.split()
        for l in textstring:
            if l.isupper():
                ucase +=1
        for t in substring:
            if t.isupper():
                upper = upper + 1
        for c in subwords:
            if c in s:
                temp = temp + [1]
            else:
                temp = temp + [0]
        for x in words:
            if x in z:
                temp = temp + [1]
            else:
                temp = temp + [0]
        temp = temp + [upper]
        temp = temp + [len(s)]
        temp = temp + [len(z)]
        temp = temp + [ucase]
        indicator_list+= [temp] 
    indicator_array = np.array(indicator_list)
    return indicator_array

def feat(series1, series2):
    terms1={}
    terms2={}
    text1 = series1.str.split()
    text2 = series2.str.split()
    for x in text1:
        for y in x:
            if len(y)>2:
                if y in terms1:
                    terms1[y] = terms1[y] + 1
                else:
                    terms1[y] = 1
    for x in text2:
        for y in x:
            if len(y)>3:
                if y in terms2:
                    terms2[y] = terms2[y] + 1
                else:
                    terms2[y] = 1
    for x in terms1:
        if x in terms2:
            terms1[x] = terms1[x] - terms2[x]
    return terms1
def selector(dictionary, number):
    newdict = {}
    for x in dictionary:
        if dictionary[x] > number:
            newdict[x] = dictionary[x]
    return newdict

def main():
    

    original_training_data = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')

    # Convert the emails to lower case as a first step to processing the text
    original_training_data['email'] = original_training_data['email'].str.lower()
    test['email'] = test['email'].str.lower()

    original_training_data['subject'].fillna("", inplace = True)

    from sklearn.model_selection import train_test_split

    train, val = train_test_split(original_training_data, test_size=0.1)
    
    subwords = list(selector(feat(ham_sub, spam_sub), 12).keys())+list(selector(feat(spam_sub, ham_sub), 4).keys())
    dictkeys = list(selector(feat(ham_mail, spam_mail), 270).keys())+list(selector(feat(spam_mail, ham_mail), 90).keys())
    X_train = wordsandre(dictkeys, subwords, train)
    Y_train = np.array(train['spam'])

    X_val = wordsandre(dictkeys, subwords, val)
    Y_val = np.array(val['spam'])
    model = LogisticRegression()
    model.fit(X_train, Y_train)

    training_accuracy = model.score(X_val, Y_val )
    print("Training Accuracy: ", training_accuracy)

    from sklearn.linear_model import LogisticRegression
    X_val = wordsandre(dictkeys, subwords, val)
    Y_val = np.array(val['spam'])
    model = LogisticRegression()
    model.fit(X_train, Y_train)

    training_accuracy = model.score(X_val, Y_val )
    print("Training Accuracy: ", training_accuracy)

    X_test = wordsandre(dictkeys, subwords, test)
    test_predictions = model.predict(X_test)

    # Construct and save the submission:
    submission_df = pd.DataFrame({
        "Id": test['id'], 
        "Class": test_predictions,
    }, columns=['Id', 'Class'])
    timestamp = datetime.isoformat(datetime.now()).split(".")[0]
    submission_df.to_csv("submission_{}.csv".format(timestamp), index=False)

    print('Created a CSV file: {}.'.format("submission_{}.csv".format(timestamp)))
    print('You may now upload this CSV file to Kaggle for scoring.')

main()

