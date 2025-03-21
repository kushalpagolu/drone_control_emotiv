from threading import Thread, Event
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from realtime_visualizer_2D import RealtimeEEGVisualizer
from kalman_filter import KalmanFilter
from emotiv_streamer_rl import EmotivStreamer
from realtimelearning_rlagent import train_agent, save_model
import time
import os

# Flag to signal the end of data collection and saving
stop_saving_thread = Event()

def save_data_continuously(data_store, filename_prefix="eeg_gyro"):
    """
    Continuously saves data to an Excel file with a timestamped filename.
    Args:
    data_store (list): List containing dictionaries of data to be saved.
    filename_prefix (str): Prefix for the filename.
    """
    while not stop_saving_thread.is_set():
        if data_store:
            filename = os.path.join("data",
                                     f"{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")  # Save in "data" folder
            try:
                df = pd.DataFrame(data_store)
                df.to_excel(filename, index=False)
                print(f"Data saved to {filename}")
                data_store.clear()  # Clear after saving to avoid duplication
            except Exception as e:
                print(f"Error saving data to Excel: {str(e)}")
        time.sleep(10)  # Save every 10 seconds to avoid excessive file writing

def start_data_collection(emotiv, visualizer, kalman_x, kalman_y, channel_names, use_drone=False):
    """
    Starts the data collection process and animates the visualization.
    """
    def data_generator():
        while True:
            packet = emotiv.read_packet()
            if packet is None:
                print("Warning: No data received from Emotiv!")
                time.sleep(0.1)  # Prevent busy-waiting
                continue  # Skip iteration

            # Apply Kalman filter
            filtered_gyro_x = kalman_x.update(packet['gyro_x'])
            filtered_gyro_y = kalman_y.update(packet['gyro_y'])

            # Save data for each new packet
            data_entry = {
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "gyro_x": filtered_gyro_x,
                "gyro_y": filtered_gyro_y,
            }

            for i, channel_name in enumerate(channel_names):
                data_entry[channel_name] = packet[channel_name]

            emotiv.data_store.append(data_entry)

            # Update visualizer buffers
            for i, channel_name in enumerate(channel_names):
                visualizer.data_buffers[i].append(packet[channel_name])

            visualizer.update_gyro_data(filtered_gyro_x, filtered_gyro_y)

            if use_drone:
                processed_eeg = emotiv.preprocess_eeg_data(data_entry)  # Preprocess EEG data
                action = train_agent(processed_eeg)
                print(f"Drone action: {action}")

            yield None  # Ensure generator continues

    # Set up the animation
    ani = FuncAnimation(
        visualizer.fig,
        visualizer.update,
        frames=data_generator,
        interval=50,  # Stream data every 50 milliseconds
        cache_frame_data=False
    )

    plt.show()

if __name__ == "__main__":
    emotiv = EmotivStreamer()
    visualizer = RealtimeEEGVisualizer()
    kalman_x, kalman_y = KalmanFilter(), KalmanFilter()
    channel_names = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
    if emotiv.connect():
        try:
            # Start the data-saving thread
            save_thread = Thread(target=save_data_continuously, args=(emotiv.data_store,))
            save_thread.daemon = True  # Allow main thread to exit even if this thread is running
            save_thread.start()

            # Start the data collection and visualization
            start_data_collection(emotiv, visualizer, kalman_x, kalman_y, channel_names)
        except KeyboardInterrupt:
            print("Session terminated.")
        finally:
            # Stop the saving thread
            stop_saving_thread.set()
            time.sleep(1)  # Give the saving thread a chance to finish
            # Ensure data is saved once more before closing
            save_thread.join(timeout=5)
            # Close the Emotiv device
            if emotiv.device:
                emotiv.disconnect()
            print("Connection closed.")
    else:
        print("Failed to connect to Emotiv device.")
