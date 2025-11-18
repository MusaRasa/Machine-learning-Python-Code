import math as mt
def sum(Num1,Num2):
    return Num1+Num2
def sub(Num1,Num2):
    return Num1-Num2
def mul(Num1,Num2):
    return Num1*Num2
def div(Num1,Num2):
    return Num1/Num2
def power(Num1,Num2):
    return Num1**Num2
def mod(Num1,Num2):
    return Num1%Num2
def trangle_calculator():
    while True:
        print("Please Enter a trangle function ")
        try:
            Trangle = int(input(" 1)Sin \n 2)Cosine \n 3)Tangent \n 4)Exit \n "))
            trangle_value = int(input("Enter the trangle value: "))
            x_trangle = mt.radians(trangle_value)
        except ValueError:
            print("You have value Error")
        if Trangle == 1:
            sinValue = mt.sin(x_trangle)
            print(sinValue)
        elif Trangle == 2:
            cosValue = mt.cos(x_trangle)
            print(cosValue)
        elif Trangle == 3:
            tanValue = mt.tan(x_trangle)
            print(tanValue)
        elif Trangle == 4:
            print("Thank You")
            break
def Simpleclaculator(): #""" In this place we have a Simple calculator"""
    while True:
        try:
            Num1 = float(input("Enter the first number: "))
            Num2 = float(input("Enter the second number: "))
            operator = input("(+)\n(-)\n(*)\n(/)\n(**)\n(%)\n ")
        except ValueError:
             print("Please enter a valid number")
             continue
        if operator == "+":
            print(f"Number1 is {Num1} and Number2 is {Num2} = {sum(Num1,Num2)}")
        elif operator == "-":
            print(f"Number1 is {Num1} and Number2 is {Num2} = {sub(Num1,Num2)}")
        elif operator == "*":
            print(f"Number1 is {Num1} and Number2 is {Num2} = {mul(Num1,Num2)}")
        elif operator == "/":
            try:
                print(f"Number1 is {Num1} and Number2 is {Num2} = {div(Num1,Num2)}")
            except ZeroDivisionError: #""" if User do a number by zero computer give error for user """
                print("you cannot divide any number to zero \n Please enter a valid number")
        elif operator == "**":
            print(f"Number1 is {Num1} and Number2 is {Num2} = {power(Num1,Num2)}")
        elif operator == "%":
            print(f"Number1 is {Num1} and Number2 is {Num2} = {mod(Num1,Num2)}")
        elif operator == "E" or operator == "0":
            print("End Of our program ")
            break
        else:
            print("Please enter a valid operator ")
try:
    name = input("Enter your name: ")
    choice = int(input("1)Trangle_calculator \n 2)Simpleclaculator \n "))
    if choice == 1:
       trangle_calculator()
    elif choice == 2:
        Simpleclaculator()
    else:
        print("Mr {} YOU Entered an invalid choice ".format(choice))
except ValueError:
    print("You have value error! Please enter a valid number ")

