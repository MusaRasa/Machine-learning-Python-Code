number = int(input("Enter a number: "))
if number<0:
   print("this is a negative number")
else:
    result = 1
    for i in range(1,number+1):
      result = result * i

      print(f"{i} , result = {result}")
