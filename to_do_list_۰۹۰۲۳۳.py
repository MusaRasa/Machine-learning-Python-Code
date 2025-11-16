Student_list = []
tri = True
while tri:
    num = int(input("you must add some Student, How many Student you want to add:  "))
    while num:
        Student = input("Enter your Student name:  ")
        Student_list.append(Student)
        print(f"{Student} is Added. \n")
        num -= 1
        if num == 0:
            break
        
    choice = int(input("1==> delete Student List .2==> show Student List: "))
   
    if choice ==1:
        print("WE have Saved these list for stdent",Student_list)
        num = int(input("How many Student you want to delet? "))
        while num:
        
            re_ta = input("Enter the Student which you want to Remove:  ")
            Student_list.remove(re_ta)
            print("This is your stored Student Name ",Student_list)
            num -=1
            if num == 0:
                ask = int(input("Do you want to continue? (1,0)")) 
                if ask == 0:
                    print("\n ...... End Of Our ProGram ...... ")
                    break
                   
                else:
                    print("Continue ...")
    elif choice == 2:
        print("We Stored these Student Name ",Student_list)
        ask = int(input("Do you want to continue? (1,0)")) 
        if ask == 0:
            print("\n ...... End Of Our ProGram ...... ")
            break
        else:
            print("Continue ...")
    else:
        print("it is a Worng Number")
    
     