# import serial 
# import time
# esp = serial.Serial('/dev/tty.usbserial-110', 9600, timeout=.1)

# def write_read(x): 
#     esp.write(bytes(x, 'utf-8'))
#     time.sleep(0.05)
#     data = esp.readline()
#     return data

# while True: 
#     num = input("Enter a number: ") # Taking input from user
#     value = write_read(num)
#     print(value) # printing the value

import requests
x = 0
while x >= 0: 
    x = int(input("Enter a number: "))
    response = requests.post("http://100.70.9.64/data", json={"cmd": "set_value", "value": x, "led": False})
    print(response)