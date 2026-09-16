from pycaw.pycaw import AudioUtilities


devices = AudioUtilities.GetSpeakers()

print("Speaker device found:")
print(devices)

print("\nAvailable device properties:")

try:
    print(devices.EndpointVolume)
except Exception as error:
    print("EndpointVolume not available:", error)