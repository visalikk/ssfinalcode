rm hypothetical_output1.txt 

rm hypothetical_test1.txt 

echo "Please enter your hypothetical place:"

read in 


echo $in>>hypothetical_test1.txt


python SentimentTest-place.py


echo "Your place:" >> hypothetical_output1.txt
 cat hypothetical_test1.txt >> hypothetical_output1.txt

echo  "-------------------------------------------------------------------------------------------" >> hypothetical_output1.txt

echo "If you go now," >> hypothetical_output1.txt

echo "Crimerate is " >> hypothetical_output1.txt

python mainModel.py Day0.csv 1 Day0_pred.csv Day0.png hypothetical_sentiment1.csv hypothetical_output1.txt 1000 1



gsutil cp -r Day0_pred.csv gs://argon-fx-191617


gsutil cp -r Day0_pred.csv gs://sentifire1.appspot.com

clear


cat hypothetical_output1.txt


