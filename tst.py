import time
import pyautogui
import keyboard

def main():
    time.sleep(2)
    while True:
        x, y = pyautogui.position()

        pyautogui.click(x, y, clicks=1, interval=0.0001)  

        # print('yes')
        
        if keyboard.is_pressed('esc'):
            break

main()