name_list = {}
def inputs():
    try:
        ask = int(input("How many Student you want to add: "))
        while ask>=1:
            name = input("Enter your name  ")
            phone_number = input(f"{name}: Enter your Phone Number: ")
            if len(phone_number)==10 and phone_number.isdigit():
                name_list[name] = phone_number
                print(f"Correctly! {name} is Added. ")
                print("_ _ _ _ _ _ _ _ _ _ _")
                ask -=1
                if ask<1:
                    print(f"Correctly these are added.")
                    break

            else:
                print("You Must Enter 10 Digit number! ")
                continue
    except ValueError:
        print("You have ValueError to your Code! ")

           
def search():
    if len(name_list) == 0:
        print("You have not entered any phone number. ")
    else:

        sname = input("Enter the Student Name which you want to Find. ")
        if sname in name_list:
            try:
                Phone_number =   input("What is (her/his) Phone Number: ")
                if Phone_number == name_list[sname]:
                    print(f"Correct! his name is {sname},Phone_Number{Phone_number}")
                else:
                    print("In Correct! ")
            except ValueError:
                 print("You have ValueError to your Code! ")
        else:
            print("We dont have any Student in this name: \n Sorry!")

def Show():
    if len(name_list) == 0:
        print("You have not entred any phone number. ")
    else:
     print(f"it is your List Which you Stored. \n{name_list}")


while True:
    try:
       choice  = int (input(" Menu\n 1==> add Student.\n 2==> search Student.\n 3==> Show StudentName.\n 0==>Exit: "))
    except ValueError:
        print("please Enter a value Number. ")
        continue
    if choice == 1:
        inputs()
    elif choice == 2:
        search()
    elif choice == 3:
        Show()
    elif choice == 0:
        print(" End of our Program! ")
        break    

    