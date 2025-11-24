from pandas import DataFrame
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer


Email = {
    'data':[
     'Hello dear you prize 100000$ ',
     'today we have an exam',
     'your id card is 2xxxxx3423 you must send money for me!',
     'you have 340 AFG to your Mobile card',
     'Afghanistan is the heart of Asia ',
     'at least you can do your home work for every day',
     'Hello how are you',
     'Please Click here and prize win!',
     'Win a free iPhone now!',
     'Meeting at 10am tomorrow',
     'Congratulations! You won $1000',
     'Please find the report attached',
     'Get cheap meds online',
     'Lets catch up over lunch',
     'Claim your free vacation',
     'Don not forget the team call',
     'Earn money from home fast',
     'Are you joining the meeting?',
     'your name is musa',
     'you can win this race if you have very toil',
        'to this cite you can win race without any skill'
    ],
   'label':[1,0,1,1,0,0,0,1,1,0,1,0,1,0,1,0,1,0,0,0,1]
}
df = DataFrame(Email)
print(df)
vectorizer = CountVectorizer() # تبدیل متن به اعداد
X = vectorizer.fit_transform(Email['data'])
y = df['label']
print(X.shape)
# Normalize
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 42)

model = MultinomialNB()
model.fit(x_train,y_train)
y_pred_test = model.predict(x_test)
acc_test = accuracy_score(y_test,y_pred_test)
print("Accuracy_score: ",acc_test*100)
re_test = recall_score(y_test,y_pred_test)
print("Recall_score: ",re_test*100)
pre_test = precision_score(y_test,y_pred_test)
print("Precision_score: ",pre_test*100)
def massage_():
    spam_massage = []
    massage = input("Enter the massage which came to your email: ")
    spam_massage.insert(0,massage)

    print("New message_ ")
    for data in spam_massage:
        message_vector = vectorizer.transform([data])
        prediction = model.predict(message_vector)[0]

        result = "ham" if prediction == 0 else "spam"
        print(f"{data} : {result}")

while True:
    massage_()

    try:
        again = input("Do you want to continue? (y/n): ")
        if again == 'y':
            print("Continue! ")
        elif again == 'n':
            print("Bye! ")
            break
            else:
                print ('you entered an invalid word. ')
    except ValueError:
        print("Please enter y or n")
