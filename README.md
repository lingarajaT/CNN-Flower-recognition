# CNN-Flower-recognition
Flower Recognition using CNN 

Required Python 3.8.10 64-bit 
Used Spyder IDE 5.5.0

Data set can be imported using link https://drive.google.com/drive/folders/1rzEFpraXXLLIGPo6F6jNEpeMOdWMgOzN?usp=sharing

PIP Installation Guide

Step 1: Download PIP get-pip.py
Before installing PIP, download the get-pip.py file. Run the following cURL command in the command prompt:


curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py

Note: if permission denied, check running as administrator

Step 2: Installing PIP on Windows
To install PIP, run the following Python command:

python get-pip.py

C:\python-3.12.1\Scripts


Edit the file python312._pth which present in Python folder, and append Lib\site-packages

Step 3: Verify Installation
To test whether the installation was successful, type the following command:

python -m pip help


Step 4: Add Pip to Path
To run PIP from any location and as a standalone command, add it to Windows environment variables. Doing so resolves the "not on Path" error.

To add PIP to Path, follow these steps:

1. Open the Start menu, search for Environment Variables, and press Enter.



Step 5: Configuration
In Windows, the PIP configuration file can be found in several locations. To view the current configuration and list all possible file locations, use the following command:

pip config -v list

Step 6: pip3 install opencv-python
