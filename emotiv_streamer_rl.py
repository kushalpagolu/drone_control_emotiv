import hid
import numpy as np
from Crypto.Cipher import AES
import struct
from datetime import datetime
import socket
import time
from scipy.signal import welch
from sklearn.decomposition import PCA

class EmotivStreamer:

    def __init__(self):
        self.vid = 0x1234
        self.pid = 0xed02
        self.device = None
        self.cipher = None
        self.cypher_key = bytes.fromhex("31003554381037423100354838003750")
        self.filename = f"eeg_gyro_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        self.data_store = []
        self.channel_names = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
        self.sampling_rate = 128  # Hz, typical for EPOC+
        self.pca = PCA(n_components=5)  # Reduce to 5 components after band power
        self.pca_fitted = False #Flag to check if pca is already fitted

    def connect(self):
        """Connects to the Emotiv EPOC+ device."""
        try:
            self.device = hid.device()
            self.device.open(self.vid, self.pid)

            if self.device is None:
                print("Error: Device object is None after opening.  Check VID/PID or permissions.")
                return False

            print(f"Connected to Emotiv device {self.vid:04x}:{self.pid:04x}") #Confirm connection
            self.device.set_nonblocking(1)
            self.cipher = AES.new(self.cypher_key, AES.MODE_ECB)
            return True
        except Exception as e:
            print(f"Connection failed: {str(e)}")
            return False

    def disconnect(self):
        """Disconnects from the Emotiv EPOC+ device."""
        if self.device:
            self.device.close()
            print("Disconnected from Emotiv device")

    def read_packet(self):
        """Reads and decrypts a packet of data from the Emotiv EPOC+."""
        channel_names = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
        try:
            try:
                encrypted = bytes(self.device.read(32))
            except Exception as e:
                print(f"Error reading from device: {e}")
                return None

            if not encrypted:
                print("No data received from Emotiv device!")
                return None

            #print(f"Encrypted data: {encrypted.hex()}") # Debugging line

            decrypted = self.cipher.decrypt(encrypted)

            # Ensure we have enough data to unpack EEG and other values
            if len(decrypted) < 32:
                print(f"Invalid packet received. Length: {len(decrypted)}")
                return None

            #print(f"Decrypted data: {decrypted.hex()}") # Debugging line

            eeg_data = [int.from_bytes(decrypted[i:i+2], 'big', signed=True) for i in range(1, 29, 2)]

            # Check if all EEG channels are properly received
            if len(eeg_data) != len(channel_names):
                print(f"EEG data missing or corrupted! Expected {len(channel_names)} channels, but got {len(eeg_data)}.")
                return None

            data = {
                'timestamp': datetime.now().isoformat(),
                'counter': decrypted[0],
                'gyro_x': decrypted[29] - 102,
                'gyro_y': decrypted[30] - 204,
                'battery': (decrypted[31] & 0x0F)
            }

            for i, channel_name in enumerate(channel_names):
                data[channel_name] = eeg_data[i] # Assign EEG data to channels

            return data

        except Exception as e:
            print(f"Error reading packet: {e}") # More specific error
            return None

    def calculate_band_power(self, eeg_data):
        """Calculates band power features (Delta, Theta, Alpha, Beta) for each channel."""
        band_power = {}
        for channel in self.channel_names:
            f, psd = welch(eeg_data[channel], fs=self.sampling_rate, nperseg=256)  # Welch's method
            band_power[channel] = {
                'Delta': np.sum(psd[(f >= 1) & (f <= 4)]),
                'Theta': np.sum(psd[(f > 4) & (f <= 8)]),
                'Alpha': np.sum(psd[(f > 8) & (f <= 12)]),
                'Beta':  np.sum(psd[(f > 12) & (f <= 30)])
            }
        return band_power

    def preprocess_eeg_data(self, data):
        """
        Processes the EEG data by:
        1. Calculating band power features.
        2. Applying PCA for dimensionality reduction.
        """
        band_power_data = self.calculate_band_power(data)

        # Flatten band power features into a single array
        flattened_features = []
        for channel in self.channel_names:
            flattened_features.extend([band_power_data[channel]['Delta'],
                                       band_power_data[channel]['Theta'],
                                       band_power_data[channel]['Alpha'],
                                       band_power_data[channel]['Beta']])

        eeg_features = np.array(flattened_features).reshape(1, -1)  # Reshape for PCA

        # Apply PCA
        if not self.pca_fitted:
            self.pca.fit(eeg_features)  # Fit PCA on the first data received
            self.pca_fitted = True
        pca_transformed = self.pca.transform(eeg_features)

        # Combine PCA components with gyro data
        processed_data = np.concatenate(([data['gyro_x'], data['gyro_y']], pca_transformed[0]))

        return processed_data

    def stream_data_to_rl_agent(self, eeg_data, host="localhost", port=12345):
        """Streams the preprocessed EEG data to the RL agent."""
        processed_data = self.preprocess_eeg_data(eeg_data)
        message = processed_data.tobytes()  # Convert to binary format for transmission
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.sendto(message, (host, port))  # Send to RL agent

    def run(self):
        """Main function to connect, read data, and stream to RL agent."""
        if not self.connect():
            return

        try:
            while True:
                eeg_data = self.read_packet()
                if eeg_data:
                    #print(f"Sending data to RL agent: {eeg_data}")  # Reduced verbosity
                    self.stream_data_to_rl_agent(eeg_data)
                time.sleep(0.05)  # Adjusted sleep time
        except KeyboardInterrupt:
            print("Streaming stopped by user")
        finally:
            self.disconnect()

if __name__ == "__main__":
    streamer = EmotivStreamer()
    streamer.run()
