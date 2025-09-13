import genomix
import os


g = genomix.connect('localhost')
g.rpath(os.environ['HOME'] + '/dvl/lib/genom/pocolibs/plugins')
rotorcraft = g.load('rotorcraft')

print('ready...')

def setup():
    rotorcraft.connect({'serial': '/dev/ttyACM0', 'baud': 57600})
    rotorcraft.set_sensor_rate({'rate': {'imu': 1000, 'mag': 0, 'motor': 20, 'battery': 1}})
    rotorcraft.set_imu_filter({'gfc': [20, 20, 20], 'afc': [5, 5, 5], 'mfc': [20, 20, 20]})
    rotorcraft.log('/home/rpi/dvl/robotpkg/logs/rotorcraft.log')
    print('setup complete')

# def set_velocities_loop():
#     velocities = [0, 0, 0, 0, 20, 30, 25, 30]
#     while True:
#         try:
#             rotorcraft.set_velocity(velocities)
#         except Exception as e:
#             print(f"Error sending velocity command: {e}")
#         await asyncio.sleep(0.02)  # 50Hz

def set_velocities():
    velocities = [0, 0, 0, 0, 20, 30, 25, 30]
    print('setting velocities')
    rotorcraft.set_velocity(velocities)
    print('set velocities')
    rotorcraft.start()
    print('rotorcraft started')


# def main():
    # setup()
    # rotorcraft.start()
    # time.sleep(1)

    # Removed rotorcraft.stop() to allow velocity commands to take effect
    # await set_velocities_loop()


# if __name__ == '__main__':
#     asyncio.run(main())

# def stop():
#     rotorcraft.stop()
#     rotorcraft.log_stop()

setup()