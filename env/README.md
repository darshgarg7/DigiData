# Environment Setup

This set up guide is mainly based on MacOS. We expect the setup to be similar on Windows and Linux.

## Prerequisites:
- Android Studio: Install [Android Studio](https://developer.android.com/studio) on your MacBook to create and manage Android emulators.
- Python: Ensure you have Python v3.12 installed.
- Node.js: Ensure you have [Node.js](https://nodejs.org/en/) installed.
- pip: The package installer for Python, comes bundled with Python.

## Installation
### Step 1: Set up Android Emulator
Open Android Studio and create a new virtual device by going to Tools -> Device Manager -> Add a new device and make sure the configuration is as follows:

- Device Definition: Pixel 7
- System Image: API 35
- Google Play: Checked
- AVD Name: DigiData

After the emulator is created, click on the play button to start the emulator. Note that in order to run the benchmarks, you will need to install all the required Apps specified in the task definitions, e.g. [task_registry/demo_3.jsonl](../task_registry/demo_3.jsonl) in the emulator. This is a one-time setup and you can skip this step if you already have the required Apps installed.

### Step 2: Install Appium
Open Terminal on your MacBook.
Run `npm install -g appium` to install Appium globally.

### Step 3: Install Python Client Library
Run `pip install Appium-Python-Client` in Terminal to install the Appium Python client library.

### Step 4: Install Appium driver

Run `appium driver install uiautomator2` in Terminal to install the Appium UIAutomator2 driver.

At this point, all the required dependencies should be installed and you should be able to set up the environment for running the benchmarks. Follow the next section to start all the required services.

## Start All services

In order to run the benchmarks, you need to start both the emulator and Appium server. Follow the steps below to start all the services:

### Step 1: Start the Android Emulator

Open Android Studio and click on the play button to start the emulator. You can also start the emulator from the command line by running `emulator -avd DigiData "-skip-adb-auth" "-no-boot-anim" "-gpu" "auto" "-no-snapshot-load"` in Terminal.

If you only want to ge the evaluation results and do not need to see the UI, you can use window-less mode by adding the flag `"-no-window"` to the command. For example, `emulator -avd DigiData "-skip-adb-auth" "-no-boot-anim" "-gpu" "auto" "-no-snapshot-load" "-no-window"`

### Step 2: Start Appium Server

Open a new Terminal window and run `appium --relaxed-security` to start the Appium server. Keep it open all the time while running the benchmarks.






