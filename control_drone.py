from djitellopy import Tello
import time
def control_drone(action):
    tello = Tello()
    try:
        tello.connect()
        tello.takeoff()
        
        if action == 0:  # Forward
            tello.move_forward(30)
        elif action == 1:  # Backward
            tello.move_back(30)
        elif action == 2:  # Left
            tello.move_left(30)
        elif action == 3:  # Right
            tello.move_right(30)
        
        time.sleep(1)  # Wait for the action to complete
        tello.land()
    except Exception as e:
        print(f"Error controlling drone: {e}")
    finally:
        tello.end()

# Example usage:
# control_drone(0)  # Move forward
