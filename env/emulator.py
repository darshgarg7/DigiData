# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import subprocess
from appium import webdriver
from appium.options.android import UiAutomator2Options
from PIL import Image
import asyncio

def get_running_emulators():
    # Run the adb devices command
    result = subprocess.run(['adb', 'devices'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Check for errors
    if result.returncode != 0:
        print("Error running adb command:", result.stderr)
        return []
    # Parse the output
    lines = result.stdout.strip().split('\n')
    emulators = []
    
    # Skip the first line as it is a header
    for line in lines[1:]:
        if line.strip():
            parts = line.split()
            if len(parts) > 1 and parts[1] == 'device':
                emulators.append(parts[0])
    
    return emulators


class Emulator():
    """
    This class is a wrapper for the Appium client. It provides methods for interacting with the emulator.
    """
    def __init__(self, config={}):
        self.temp_dir = config.get('temp_dir', './')
        self.appium_server_url = config.get('appium_server_url', "http://0.0.0.0:4723")
        self.driver = None
        self.screen_size = None
    
    def connect(self, device_name, udid):
        capabilities = dict(
            platformName='Android',
            automationName='uiautomator2',
            deviceName=device_name,
            newCommandTimeout="120000",
            adbExecTimeout="120000",
            noReset=True,
            uiautomator2ServerInstallTimeout="120000",
            uiautomator2ServerLaunchTimeout="120000",
            uiautomator2ServerReadTimeout="120000",
            udid=udid
        )
        
        options = UiAutomator2Options().load_capabilities(capabilities)
        self.driver = webdriver.Remote(self.appium_server_url, options=options)
        screen_size = self.driver.get_window_size()
        self.screen_size = screen_size

    def get_screenshot(self):
        temp_screenshot_filepath = os.path.join(self.temp_dir, 'screenshot.png')
        if self.driver:
            self.driver.save_screenshot(temp_screenshot_filepath)
            return Image.open(temp_screenshot_filepath).convert('RGB')
        else:
            raise Exception('Emulator is not connected, please call connect() first')

    def get_xml(self):
        ui_raw = self.driver.page_source
        return ui_raw

    def get_screen_size(self):
        return self.screen_size

    def run_adb_command(self, command):
        # print(f"Running adb command: {command}")
        if self.driver:
            return self.driver.execute_script('mobile:shell', {'command': command})
        else:
            raise Exception('Emulator is not connected, please call connect() first')

    def execute_action(self, action):
        match action.function_name:
            case 'tap':
                [x, y] = action.args[0]
                self.run_adb_command(f'input tap {x} {y}')
            case 'swipe':
                [x1, x2] = action.args[0]
                [y1, y2] = action.args[1]
                self.run_adb_command(f'input swipe {x1} {y1} {x2} {y2}')
            case 'type':
                text = action.args[0]
                self.run_adb_command(f'input text "{text}"')
            case 'navigate(home)':
                self.run_adb_command('input keyevent 3')
            case 'navigate(back)':
                self.run_adb_command('input keyevent 4')
            case 'complete':
                pass
            case _:
                raise Exception(f'Unknown action: {action.function_name}')


class LocalEmulatorPool:
    """
    Manages a pool of locally connected emulators.
    """
    def __init__(self, device_name='DigiData'):
        emulator_ids = get_running_emulators()
        if not emulator_ids:
            raise Exception("No running emulators found")
        self.pool = asyncio.Queue()
        for udid in emulator_ids:
            emu = Emulator()
            emu.connect(device_name, udid)
            self.pool.put_nowait(emu)

    async def acquire(self):
        return await self.pool.get()

    def release(self, emulator):
        self.pool.put_nowait(emulator)