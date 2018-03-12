import matplotlib
matplotlib.use('Agg')
import math
import itertools

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import sys
import csv
import numpy as ny
import pandas
import matplotlib.pyplot as plt

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)



#X = ny.array([[1,2], [3,4], [5,6], [7,8], [9,10]])
#Y = ny.array([3, 4, 5, 6, 7])

#Filename = sys.argv[1]
#inputsize = int(sys.argv[2])
#TrainOutputFile=sys.argv[3]
#PlotFile=sys.argv[4]
#TestInputFile=sys.argv[5]
#TestOutputFile=sys.argv[6]
#trials=int(sys.argv[7])
#verboseVal=int(sys.argv[8])

Filename = 'Day0.csv'
inputsize = 1
TrainOutputFile='Day0_Pred.csv'
PlotFile=Day0.png
TestInputFile='NA'
TestOutputFile='NA'
trials=1000
verboseVal=1

X = ny.empty([0,inputsize])
Y = ny.empty([0,1])


with open(Filename) as csvfile:
    readCSV=csv.reader(csvfile,delimiter=',')
    for row in readCSV:
        X = ny.vstack([X,[row[:inputsize]]])
        Y = ny.vstack([Y,row[inputsize]])
#print X
#print Y

sc_X=StandardScaler()
X_train = sc_X.fit_transform(X)

Y=ny.reshape(Y,(-1,1))
sc_Y=StandardScaler()
Y_train = sc_Y.fit_transform(Y)


N = len(Y_train)

def brain():
    #Create the brain
    br_model=Sequential()
    br_model.add(Dense(40, input_dim=inputsize, kernel_initializer='normal',activation='relu'))
    br_model.add(Dense(30, kernel_initializer='normal',activation='relu'))


    br_model.add(Dense(20, kernel_initializer='normal',activation='relu'))

    br_model.add(Dense(20, kernel_initializer='normal',activation='relu'))
  
    br_model.add(Dense(15, kernel_initializer='normal',activation='relu'))

    br_model.add(Dense(1,kernel_initializer='normal'))
    
    #Compile the brain
    br_model.compile(loss='mean_squared_error',optimizer='adam')
#    br_model.compile(loss='logcosh',optimizer='adam')
    return br_model

def predict(X,sc_X,sc_Y,estimator):
    prediction = estimator.predict(sc_X.transform(X))
    return sc_Y.inverse_transform(prediction)



estimator = KerasRegressor(build_fn=brain, epochs=trials, batch_size=5,verbose=verboseVal)
# print "Done"


# seed = 21
# ny.random.seed(seed)
# kfold = KFold(n_splits=N, random_state=seed)
# results = cross_val_score(estimator, X_train, Y_train, cv = kfold)
estimator.fit(X_train,Y_train)
prediction = estimator.predict(X_train)

# print Y_train
# print prediction

# print Y
# pred_final= sc_Y.inverse_transform(prediction)
pred_final = predict(X,sc_X,sc_Y,estimator)
# print pred_final

X_trainOut=ny.empty([0,3])
Base = ny.empty([0,1])
errorVal=0
X_trainOut=ny.vstack([X_trainOut,['SNo','Actual','Predicted']])
for i in range(0, len(Y)):
   row_new = [i,Y[i][0], pred_final[i]]
   X_trainOut=ny.vstack([X_trainOut,row_new])
   errorVal=errorVal+pow(float(Y[i][0])-float(pred_final[i]),2)
   Base=ny.vstack([Base,i])

errorVal=errorVal/len(Y)
   
#print ('Average Deviation:')
print ('DONE:')
print (math.sqrt(errorVal))
with open(TrainOutputFile,'wb') as csvWriteFile:
    writeCSV=csv.writer(csvWriteFile,delimiter=",")
    writeCSV.writerows(X_trainOut)


if inputsize==1:
    plt.plot(X.astype(float),Y.astype(float),'rx')
# plt.xticks(ny.arange(min(X),max(X),1))
# plt.yticks(ny.arange(4.0,10.0,1.0))
# plt.savefig('plotOr.png')
# plt.clf()
    lists=sorted(itertools.izip(*[X.astype(float),pred_final.astype(float)]))
    new_x, new_y= list(itertools.izip(*lists))
    plt.plot(new_x,new_y,'g')
#    plt.plot(X.astype(float),pred_final.astype(float), 'gx')
    plt.yticks(ny.arange(min(Y.astype(float)),max(Y.astype(float)),(max(Y.astype(float))-min(Y.astype(float)))/10))
    plt.savefig(PlotFile)
#plt.clf()
#plt.plot(Y,pred_final,'gx')
#plt.savefig('plotX.png')

# print results.mean()
# print results.std()

if TestInputFile != 'NA':
#    print TestInputFile
    testdata = ny.empty([0,inputsize])
    with open(TestInputFile) as csvfile:
        readCSV=csv.reader(csvfile,delimiter=',')
        for row in readCSV:
            testdata = ny.vstack([testdata,row[:inputsize]])
#    print testdata
#    testdata=ny.reshape(testdata,(-1,1))

    prediction = estimator.predict(sc_X.transform(testdata))
#    print prediction
    testprediction= ny.asarray(sc_Y.inverse_transform([float(prediction)]))

#    print testprediction
#    with open(TestOutputFile,'wb') as the_File:
#        the_Filewriter(csvWriteFile,delimiter=",")
#        writeCSV.writerows(testprediction)
    temp = testprediction[0]
#    print str(temp)
    with open(TestOutputFile, 'a') as the_file:
        the_file.write(str(temp))
        the_file.write('\n')
