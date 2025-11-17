# steps
# 1)listed opetion =>s:sang,k:kaghaz,:qeiche
# 2)Usser choose
# 3)choose pc -->rendom
# 4)compare ==> how is winner?
# 5)repeat



import random 
option = ["s","k","q"]
pc_s=0
user_s =0
run = True

while run:
    
    usser_choice = input("Choose an opetion ==> s:stone, p:peaper, q:qaiche \n")
    pc_choice = random.choice(option)
    print(f"Pc choice {pc_choice}")
    if usser_choice == pc_choice:
      print(" Equal ")
    
    elif usser_choice == "s":
        if pc_choice =="k":
            print("pc win!")
            pc_s +=1
        else:
           print("Usser Win!")
           user_s += 1
    #   tri+=1
    elif usser_choice == "k":
        if pc_choice == "q":
            print("Pc_win")
            pc_s +=1
        else:
            print("user_win! ")
            user_s += 1
    elif usser_choice == "q":
        if pc_choice == "s":
            print("Pc Win!")  
            pc_s +=1
        else:
            print("User Win! ")
            user_s += 1
    print(f"Continue! user_s {user_s}; pc_s:{pc_s}")        
    if pc_s == 3 or user_s ==3:
        if pc_s == 3:
            print("Pc_win!!!!!!!!!!!")
        else:
            print("User_Win!!!!!!!!!")
            
        run=False
    
# problems
# 1) options["stone_paper_Scissors"]
# 2) pc_chooice = rendom.choice
