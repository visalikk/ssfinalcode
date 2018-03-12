rm hypothetical_output.txt 


rm hypothetical_test.txt


echo "Please enter your hypothetical comments:"


read in 


echo $in>>hypothetical_test.txt



python SentimentTest.py



echo "Your text:" >> hypothetical_output.txt
cat hypothetical_test.txt >> hypothetical_output.txt

echo  "-------------------------------------------------------------------------------------------" >> hypothetical_output.txt

echo "If your event were to happen today," >> hypothetical_output.txt
echo "Today's prediction (compared to history, in %):" >> hypothetical_output.txt

python mainModel.py Day0.csv 1 Day0_pred.csv Day0.png hypothetical_sentiment.csv hypothetical_output.txt 1000 1



gsutil cp -r Day0_pred.csv gs://argon-fx-191617

 
gsutil cp -r Day0_pred.csv gs://sentifire1.appspot.com

clear

cat hypothetical_output.txt


