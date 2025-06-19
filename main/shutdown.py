from gpiozero import Button
import time
import os

# 用 GPIO 4（物理引脚 7）
button = Button(4, hold_time=3, pull_up=True)

def handle_press():
    print("Button held down")
    press_time = time.time()
    while button.is_held:
        time.sleep(0.1)
    duration = time.time() - press_time
    print(f"Held for {duration:.2f} seconds")

    if duration >= 5:
        print("Rebooting system...")
        os.system("sudo /sbin/reboot")
    elif duration >= 3:
        print("Shutting down system...")
        os.system("sudo /sbin/shutdown -h now")
    else:
        print("Press too short, ignored.")

while True:
    print("Waiting for button press...")
    button.wait_for_press()
    if button.is_held:
        handle_press()
