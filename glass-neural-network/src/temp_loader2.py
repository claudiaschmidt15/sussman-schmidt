#### Libraries
# Standard library
import cPickle
import gzip

# Third-party libraries
import numpy as np


def load_data():
    #with open('.../data/TO.45_train_data.pkl','rb') as f:
        #training_data, validation_data, test_data = cPickle.load(f)
    #f.close()
    #return (training_data, validation_data, test_data)
    training_data, validation_data, test_data = cPickle.load(open('TO.45_train_data.pkl.icloud', 'rb')) 


def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (100, 1)) for x in tr_d[0]]
    training_results = [training_inputs, tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (100, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (100, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)