#!/usr/bin/env python

# Copyright (c) 2019 Intel Labs
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Allows controlling a vehicle with a keyboard. For a simpler and more
# documented example, please take a look at tutorial.py.

"""
Welcome to CARLA manual control with steering wheel Logitech G29.

To drive start by preshing the brake pedal.
Change your wheel_config.ini according to your steering wheel.

To find out the values of your steering wheel use jstest-gtk in Ubuntu.

"""

from __future__ import print_function
import csv
from csv import writer
import matplotlib.pyplot as plt
import os
import time


# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================

import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================


import carla

from carla import ColorConverter as cc

import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref
import pandas as pd


if sys.version_info >= (3, 0):

    from configparser import ConfigParser

else:

    from ConfigParser import RawConfigParser as ConfigParser

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_h
    from pygame.locals import K_m
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_w
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')


# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================


def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


"""
directions of figure 8:
town  - south, mountain nort
under the bridge  - south to north 
over the bridge - east to west

start point coordinates:
south (-19.5, -226.8)


finish point:

"""


# ==============================================================================
# -- BicycleRider Class -------------------------------------------------------
# ==============================================================================
class BicycleRider:
    def __init__(self, world, traffic_manager, spawn_point):
        print("BicycleRider init")
        self.world = world
        self.traffic_manager = traffic_manager
        self.spawn_point = spawn_point
        self.actor = None
        self.spawn_bicycle()

    def spawn_bicycle(self):
        """Spawn the bicycle and configure its behavior."""
        # Define the blueprint for a bicycle
        blueprint = self.world.get_blueprint_library().find('vehicle.diamondback.century')
        if not blueprint:
            raise RuntimeError('Bicycle blueprint not found!')
        
        # Set attributes like color
        blueprint.set_attribute('color', '255, 234, 0')

        # Spawn the bicycle at the specified location
        self.actor = self.world.try_spawn_actor(blueprint, self.spawn_point)
        if self.actor:
            print(f"Bicycle spawned with ID: {self.actor.id}")
            self.configure_behavior()
        else:
            print("Failed to spawn bicycle.")

    def configure_behavior(self,change_lane=False,speed_percentage=50):
        """Configure traffic manager behavior for the bicycle."""
        if not self.actor:
            print("No actor to configure.")
            return

        # Disable lane changes
        self.traffic_manager.auto_lane_change(self.actor, change_lane)

        self.traffic_manager.vehicle_percentage_speed_difference(self.actor, speed_percentage)

        # Enable autopilot and set the speed
        self.actor.set_autopilot(True, self.traffic_manager.get_port())
        print(f"Bicycle {self.actor.id} configured with autopilot and custom behavior.")

    def set_target_velocity(self, velocity):
        """Set a custom target velocity for the bicycle."""
        if self.actor:
            self.actor.set_target_velocity(carla.Vector3D(x=velocity, y=0.0, z=0.0))
            print(f"Bicycle velocity set to {velocity} m/s.")

    def get_location(self):
        """Get the current location of the bicycle."""
        if self.actor:
            return self.actor.get_location()
        return None

    def destroy(self):
        """Destroy the actor."""
        print("Destroying bicycle.")
        if self.actor:
            print(f"Destroying bicycle with ID: {self.actor.id}")
            self.actor.destroy()
            self.actor = None


# class BicycleRider:
#     def __init__(self, world, spawn_point):
#         self.world = world
#         self.spawn_point = spawn_point
#         self.actor = None
#         self.spawn_bicycle()

#     def spawn_bicycle(self):
#         # Define the blueprint for a bicycle; update this according to your specific blueprint
#         blueprint = self.world.get_blueprint_library().find('vehicle.diamondback.century')
#         blueprint.set_attribute('color', '255, 234, 0')
#         if not blueprint:
#             raise RuntimeError('Bicycle blueprint not found!')

#         # Spawn the bicycle at the specified location
#         self.actor = self.world.spawn_actor(blueprint, self.spawn_point)
#         # self.bicycle.apply_control(carla.VehicleControl(throttle = 0, brake = 1))
#         self.actor.set_autopilot(True)  # Optionally enable autopilot
#         self.actor.set_target_velocity(25)

#     def destroy(self):
#         if self.actor:
#             self.actor.destroy()

# ==============================================================================
# -- MotorbikeRider Class -------------------------------------------------------
# ==============================================================================
class MotorbikeRider:
    def __init__(self, world, traffic_manager, spawn_point):
        print("MotorbikeRider init")
        self.world = world
        self.traffic_manager = traffic_manager
        self.spawn_point = spawn_point
        self.actor = None
        self.spawn_motorbike()

    def spawn_motorbike(self):
        """Spawn the motorbike and configure its behavior."""
        # Define the blueprint for a motorbike
        blueprint = self.world.get_blueprint_library().find('vehicle.kawasaki.ninja')
        if not blueprint:
            raise RuntimeError('Motorbike blueprint not found!')
        
        # Set attributes like color if applicable
        blueprint.set_attribute('color', '255, 0, 0')  # Example color setting

        # Spawn the motorbike at the specified location
        self.actor = self.world.try_spawn_actor(blueprint, self.spawn_point)
        if self.actor:
            print(f"Motorbike spawned with ID: {self.actor.id}")
            self.configure_behavior()
        else:
            print("Failed to spawn motorbike.")

    def configure_behavior(self,change_lane=False,speed_percentage=50):
        """Configure traffic manager behavior for the motorbike."""
        if not self.actor:
            print("No actor to configure.")
            return

        # Disable lane changes
        self.traffic_manager.auto_lane_change(self.actor, change_lane)

        # Set a speed limit (e.g., 20% below default speed)
        self.traffic_manager.vehicle_percentage_speed_difference(self.actor, speed_percentage)

        # Enable autopilot and configure behavior
        self.actor.set_autopilot(True, self.traffic_manager.get_port())
        print(f"Motorbike {self.actor.id} configured with autopilot and custom behavior.")

    def set_target_velocity(self, velocity):
        """Set a custom target velocity for the motorbike."""
        if self.actor:
            self.actor.set_target_velocity(carla.Vector3D(x=velocity, y=0.0, z=0.0))
            print(f"Motorbike velocity set to {velocity} m/s.")

    def set_heading(self, yaw):
        """Set the heading (yaw) of the motorbike."""
        if not self.actor:
            print("No actor to set heading.")
            return

        # Get current transform and update yaw
        transform = self.actor.get_transform()
        transform.rotation.yaw = yaw  # Set the yaw (heading) in degrees
        self.actor.set_transform(transform)
        print(f"Motorbike heading set to {yaw} degrees.")

    def get_location(self):
        """Get the current location of the motorbike."""
        if self.actor:
            return self.actor.get_location()
        return None

    def destroy(self):
        """Destroy the actor."""
        print("Destroying motorbike.")
        if self.actor:
            print(f"Destroying motorbike with ID: {self.actor.id}")
            self.actor.destroy()
            self.actor = None


# class MotorbikeRider:
#     def __init__(self, world, spawn_point):
#         print("MotorbikeRider init")
#         self.world = world
#         self.spawn_point = spawn_point
#         self.actor = None
#         self.spawn_motorbike()

#     def spawn_motorbike(self):
#         # Define the blueprint for a motorbike; update this according to your specific blueprint
#         blueprint = self.world.get_blueprint_library().find('vehicle.kawasaki.ninja')
#         # self.motorbike.set_attribute('color', '255, 234, 0')
#         if not blueprint:
#             raise RuntimeError('Motorbike blueprint not found!')

#         # Spawn the motorbike at the specified location
#         self.actor = self.world.spawn_actor(blueprint, self.spawn_point)
#         self.actor.set_autopilot(True)  # Optionally enable autopilot
        


#     def destroy(self):
#         if self.actor:
#             print(f"Destroying motorbike with ID: {self.actor.id}")
#             self.actor.destroy()


# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================


class World(object):
    def __init__(self, carla_world, hud, actor_filter,log,client):
        print("World init")
        self.world = carla_world
        self.client = client
        self.traffic_manager = self.client.get_trafficmanager()
        self.hud = hud
        self.player = None
        self.bicycle = None
        self.motorbike = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = actor_filter
        self.restart()
        self.world.on_tick(hud.on_world_tick)
        self.log = log

        #south (-19.5, -226.8)



    def restart(self):
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
        # Get a random blueprint.
        blueprint = random.choice(self.world.get_blueprint_library().filter(self._actor_filter))
        blueprint.set_attribute('role_name', 'hero')
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        #spwan bicycle rider
        if self.bicycle is not None:
            # spawn_point = self.bicycle.get_transform()
            spawn_point_bicycle = carla.Transform(carla.Location(x=-271.1, y=37.1, z=2),carla.Rotation(yaw=0))
            self.destroy()
            self.bicycle.actor = self.world.try_spawn_actor(blueprint, spawn_point_bicycle)
        while self.bicycle is None:
            # spawn_points = self.world.get_map().get_spawn_points()
            spawn_point_bicycle = carla.Transform(carla.Location(x=-271.1, y=37.1, z=2),carla.Rotation(yaw=0))
            self.bicycle = BicycleRider(self.world,self.traffic_manager, spawn_point_bicycle)

        #spwan motorbike rider
        if self.motorbike is not None:
            # spawn_point = self.motorbike.get_transform()
            spawn_point_motorbike = carla.Transform(carla.Location(x=-67.2, y=37.3, z=13),carla.Rotation(yaw=0))
            spawn_point_motorbike.location.z = self.world.get_map().get_spawn_points()[0].location.z
            self.destroy()
            self.motorbike.actor = self.world.try_spawn_actor(blueprint, spawn_point_motorbike)
        while self.motorbike is None:
            # spawn_points = self.world.get_map().get_spawn_points()
            spawn_point_motorbike = carla.Transform(carla.Location(x=-67.2, y=37.3, z=13),carla.Rotation(yaw=0))
            self.motorbike = MotorbikeRider(self.world,self.traffic_manager, spawn_point_motorbike)

        # Spawn the player.
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 10
            spawn_point.rotation.pitch = 10
            spawn_point.rotation.yaw = 180
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        while self.player is None:
            # spawn_points = self.world.get_map().get_spawn_points()
            spawn_point = carla.Transform(carla.Location(x=-433.9, y=34.9, z=2),carla.Rotation(yaw=0))
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    def tick(self, clock):
        self.hud.tick(self, clock)
        #    def configure_behavior(self,change_lane=False,speed_percentage=50):
        player_loc =self.player.get_location()
        bicycle_loc =self.bicycle.actor.get_location()
        motorbike_loc =self.motorbike.actor.get_location()
        if(calculate_distance(player_loc,bicycle_loc) > 100):
            self.bicycle.set_target_velocity(0)
        else:
            self.motorbike.configure_behavior(change_lane=True,speed_percentage=20)
        if(calculate_distance(player_loc,motorbike_loc) > 100):
            self.motorbike.set_target_velocity(0)
        else:
            self.motorbike.configure_behavior(change_lane=True,speed_percentage=20)



    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)
        

    def destroy(self):
        print("Destroying world")
        sensors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor]
        for sensor in sensors:
            if sensor is not None:
                sensor.stop()
                sensor.destroy()
        if self.player is not None:
            self.player.destroy()
        if self.bicycle is not None:
            self.bicycle.destroy()
        if self.motorbike is not None:
            self.motorbike.destroy()
        if self.log is not None:
            self.log.destroy()



# ==============================================================================
# -- DualControl -----------------------------------------------------------
# ==============================================================================


class DualControl(object):
    def __init__(self, world, start_in_autopilot,writer):
        print("DualControl init")
        self._autopilot_enabled = start_in_autopilot
        if isinstance(world.player, carla.Vehicle):
            self._control = carla.VehicleControl()
            world.player.set_autopilot(self._autopilot_enabled)
        elif isinstance(world.player, carla.Walker):
            self._control = carla.WalkerControl()
            self._autopilot_enabled = False
            self._rotation = world.player.get_transform().rotation
        else:
            raise NotImplementedError("Actor type not supported")
        self._steer_cache = 0.0
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

        # initialize steering wheel
        pygame.joystick.init()

        joystick_count = pygame.joystick.get_count()
        if joystick_count > 1:
            raise ValueError("Please Connect Just One Joystick")

        self._joystick = pygame.joystick.Joystick(0)
        self._joystick.init()

        self._parser = ConfigParser()
        self._parser.read(r'C:\Users\CARLA-1\Desktop\project\carla\WindowsNoEditor\PythonAPI\examples\wheel_config.ini')
        # print("sections: ",self._parser.sections())
        self._steer_idx = int(
            self._parser.get('G920 Racing Wheel', 'steering_wheel'))
        self._throttle_idx = int(
            self._parser.get('G920 Racing Wheel', 'throttle'))
        self._brake_idx = int(self._parser.get('G920 Racing Wheel', 'brake'))
        self._reverse_idx = int(self._parser.get('G920 Racing Wheel', 'reverse'))
        self._handbrake_idx = int(
            self._parser.get('G920 Racing Wheel', 'handbrake'))

    def parse_events(self, world, clock,v):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.JOYBUTTONDOWN:
                if event.button == 0:
                    world.restart()
                elif event.button == 1:
                    world.hud.toggle_info()
                elif event.button == 2:
                    world.camera_manager.toggle_camera()
                elif event.button == 3:
                    world.next_weather()
                elif event.button == self._reverse_idx:
                    self._control.gear = 1 if self._control.reverse else -1
                elif event.button == 23:
                    world.camera_manager.next_sensor()

            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_BACKSPACE:
                    world.restart()
                elif event.key == K_F1:
                    world.hud.toggle_info()
                elif event.key == K_h or (event.key == K_SLASH and pygame.key.get_mods() & KMOD_SHIFT):
                    world.hud.help.toggle()
                elif event.key == K_TAB:
                    world.camera_manager.toggle_camera()
                elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_weather(reverse=True)
                elif event.key == K_c:
                    world.next_weather()
                elif event.key == K_BACKQUOTE:
                    world.camera_manager.next_sensor()
                elif event.key > K_0 and event.key <= K_9:
                    world.camera_manager.set_sensor(event.key - 1 - K_0)
                elif event.key == K_r:
                    world.camera_manager.toggle_recording()
                if isinstance(self._control, carla.VehicleControl):
                    if event.key == K_q:
                        self._control.gear = 1 if self._control.reverse else -1
                    elif event.key == K_m:
                        self._control.manual_gear_shift = not self._control.manual_gear_shift
                        self._control.gear = world.player.get_control().gear
                        world.hud.notification('%s Transmission' %
                                               ('Manual' if self._control.manual_gear_shift else 'Automatic'))
                    elif self._control.manual_gear_shift and event.key == K_COMMA:
                        self._control.gear = max(-1, self._control.gear - 1)
                    elif self._control.manual_gear_shift and event.key == K_PERIOD:
                        self._control.gear = self._control.gear + 1
                    elif event.key == K_p:
                        self._autopilot_enabled = not self._autopilot_enabled
                        world.player.set_autopilot(self._autopilot_enabled)
                        world.hud.notification('Autopilot %s' % ('On' if self._autopilot_enabled else 'Off'))

        if not self._autopilot_enabled:
            if isinstance(self._control, carla.VehicleControl):
                self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time())
                self._parse_vehicle_wheel(v,world)
                self._control.reverse = self._control.gear < 0
            elif isinstance(self._control, carla.WalkerControl):
                self._parse_walker_keys(pygame.key.get_pressed(), clock.get_time())
            world.player.apply_control(self._control)

    def _parse_vehicle_keys(self, keys, milliseconds):
        self._control.throttle = 1.0 if keys[K_UP] or keys[K_w] else 0.0
        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            self._steer_cache += steer_increment
            # print(" self._steer_cache = " + str( self._steer_cache))

        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        # print("self._steer_cache = " + str(min(0.7, max(-0.7, self._steer_cache))))

        self._control.steer = round(self._steer_cache, 1)
        # print("self._control.steer = " + str(self._control.steer))
        self._control.brake = 1.0 if keys[K_DOWN] or keys[K_s] else 0.0
        self._control.hand_brake = keys[K_SPACE]

    def _parse_vehicle_wheel(self,v,world):
        numAxes = self._joystick.get_numaxes()
        jsInputs = [float(self._joystick.get_axis(i)) for i in range(numAxes)]
        # print (jsInputs)
        jsButtons = [float(self._joystick.get_button(i)) for i in
                     range(self._joystick.get_numbuttons())]
        # print (jsButtons)

        # Custom function to map range of inputs [1, -1] to outputs [0, 1] i.e 1 from inputs means nothing is pressed
        # For the steering, it seems fine as it is
        K1 =  3 # orig = 1 TODO steering coefficient, decide optimum
        steerCmd = K1 * math.tan(1.1 * jsInputs[self._steer_idx])
        # print(jsInputs[self._steer_idx])
        new_steer_cmd = map_steering_input_to_degrees(jsInputs[self._steer_idx])

        # print(new_steer_cmd)
        K2 = 1.6  # orig =  1.6 - TODO throttle shift value
        throttleCmd = K2 + (2.05 * math.log10(
            -0.7 * jsInputs[self._throttle_idx] + 1.4) - 1.2) / 0.92
        # print("Throtle: " +str(throttleCmd))

        speed = (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
        if throttleCmd <= 0 or speed > 80: # change speed limit
            throttleCmd = 0
        elif throttleCmd > 1:
            throttleCmd = 1
        # K3 = 1 #orig = 0.7 break shift
        brakeCmd = 1.6 + (2.05 * math.log10(
            -0.7 * jsInputs[self._brake_idx] + 1.4) - 1.2) / 0.92 
        if brakeCmd <= 0:
            brakeCmd = 0
        elif brakeCmd > 1:
            brakeCmd = 1

        self._control.steer = steerCmd
        self._control.brake = brakeCmd
        self._control.throttle = throttleCmd
        # new_data = {"steerCmd":steerCmd , "brakeCmd": brakeCmd, "throttleCmd":throttleCmd }
        filename = "./example.csv"
        world.player.apply_control(self._control)

        t = world.player.get_transform()
        vehicles = world.world.get_actors().filter('vehicle.*')
        distance = lambda l: math.sqrt((l.x - t.location.x)**2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)
        vehicles = [(distance(x.get_location()), x) for x in vehicles if x.id != world.player.id]


        # actors = world.world.get_actors().filter('vehicle.*')
        # for a in actors:
        #     if a.id == world.player.id:
        #         continue
        #     if calculate_distance(a.get_location(), t.location) > 110:
        #         a.apply_control(carla.VehicleControl(throttle = 0, brake = 1)) 
        #         a.set_autopilot(True)  # Optionally enable autopilot

        #     else:
        #         a.apply_control(carla.VehicleControl(throttle = 1, brake = 0))
        #         a.set_autopilot(True)  # Optionally enable autopilot


            
            



        world.log.log_data(steerCmd=new_steer_cmd, brakeCmd=brakeCmd, throttleCmd=throttleCmd, distance1=vehicles[0][0], distance2=vehicles[1][0],world = world, increment_tick = True)
        # pd.DataFrame(columns=['tick', 'data'])
        log_data(filename, steerCmd, brakeCmd, throttleCmd,0)
        # df_new = pd.DataFrame(new_data)
        # steerlog = pd.concat([steerlog, df_new], ignore_index=True)


        #toggle = jsButtons[self._reverse_idx]

        self._control.hand_brake = bool(jsButtons[self._handbrake_idx])

    def _parse_walker_keys(self, keys, milliseconds):
        self._control.speed = 0.0
        if keys[K_DOWN] or keys[K_s]:
            self._control.speed = 0.0
        if keys[K_LEFT] or keys[K_a]:
            self._control.speed = .01
            self._rotation.yaw -= 0.08 * milliseconds
        if keys[K_RIGHT] or keys[K_d]:
            self._control.speed = .01
            self._rotation.yaw += 0.08 * milliseconds
        if keys[K_UP] or keys[K_w]:
            self._control.speed = 5.556 if pygame.key.get_mods() & KMOD_SHIFT else 2.778
        self._control.jump = keys[K_SPACE]
        self._rotation.yaw = round(self._rotation.yaw, 1)
        self._control.direction = self._rotation.get_forward_vector()

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 24), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        t = world.player.get_transform()
        v = world.player.get_velocity()
        c = world.player.get_control()
        heading = 'N' if abs(t.rotation.yaw) < 89.5 else ''
        heading += 'S' if abs(t.rotation.yaw) > 90.5 else ''
        heading += 'E' if 179.5 > t.rotation.yaw > 0.5 else ''
        heading += 'W' if -0.5 > t.rotation.yaw > -179.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')
        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
            'Map:     % 20s' % world.world.get_map().name.split('/')[-1],
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)),
            u'Heading:% 16.0f\N{DEGREE SIGN} % 2s' % (t.rotation.yaw, heading),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (t.location.x, t.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % t.location.z,
            '']
        if isinstance(c, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', c.throttle, 0.0, 1.0),
                ('Steer:', c.steer, -1.0, 1.0),
                ('Brake:', c.brake, 0.0, 1.0),
                ('Reverse:', c.reverse),
                ('Hand brake:', c.hand_brake),
                ('Manual:', c.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear)]
        elif isinstance(c, carla.WalkerControl):
            self._info_text += [
                ('Speed:', c.speed, 0.0, 5.556),
                ('Jump:', c.jump)]
        self._info_text += [
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)]
        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']
            distance = lambda l: math.sqrt((l.x - t.location.x)**2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)
            vehicles = [(distance(x.get_location()), x) for x in vehicles if x.id != world.player.id]
            for d, vehicle in sorted(vehicles):
                # if d > 200.0:
                #     break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                self._info_text.append('% 4dm %s' % (d, vehicle_type))
                log_data("./example.csv", 0, 0, 0, d)


    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)


# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)


# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    def __init__(self, font, width, height):
        lines = __doc__.split('\n')
        self.font = font
        self.dim = (680, len(lines) * 22 + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * 22))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)


# ==============================================================================
# -- SpeedText ------------------------------------------------------------------
# ==============================================================================


class SpeedText(object):
    def __init__(self, font, width, height):
        lines = __doc__.split('\n')
        self.font = font
        self.dim = (680, len(lines) * 22 + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * 22))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)


# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)


# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================


class LaneInvasionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.hud.notification('Crossed line %s' % ' and '.join(text))

# ==============================================================================
# -- GnssSensor --------------------------------------------------------
# ==============================================================================


class GnssSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.0, z=2.8)), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude


# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    def __init__(self, parent_actor, hud):
        print("CameraManager init")
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        self._camera_transforms = [
            carla.Transform(carla.Location(x=-0.2,y=-0.2, z=1.3), carla.Rotation(yaw=0)),  # Main front view
            carla.Transform(carla.Location(x=-0.15, y=-0.4, z=1.2), carla.Rotation()),  # Close view
            carla.Transform(carla.Location(x=-0.15, y=-1.2, z=1.5), carla.Rotation(yaw=-165)),  # Left mirror view
            # carla.Transform(carla.Location(x=-0.15, y=1.2, z=1.7), carla.Rotation(yaw=160)),   # Right mirror view
            carla.Transform(carla.Location(x=-1.2, y=0, z=1.4), carla.Rotation(yaw=180,pitch=-15))       # Rear mirror view
            
        ]
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
        ]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        self.sensor_blueprints = []

        # Create blueprints for all sensors
        for item in self.sensors:
            bp = bp_library.find(item[0])
            bp.set_attribute('image_size_x', str(hud.dim[0]))
            bp.set_attribute('image_size_y', str(hud.dim[1]))
            item.append(bp)
            self.sensor_blueprints.append(bp)

        self.index = None
        self.left_camera = None
        # self.right_camera = None
        self.rear_camera = None
        self.left_surface = None
        # self.right_surface = None
        self.rear_surface = None

    def set_sensor(self, index, notify=True):
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else self.sensors[index][0] != self.sensors[self.index][0]
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[0],
                attach_to=self._parent)
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

        # Set up additional cameras
        self.left_camera = self._parent.get_world().spawn_actor(
            self.sensor_blueprints[0], self._camera_transforms[2], attach_to=self._parent)
        # self.right_camera = self._parent.get_world().spawn_actor(
        #     self.sensor_blueprints[0], self._camera_transforms[3], attach_to=self._parent)
        self.rear_camera = self._parent.get_world().spawn_actor(
            self.sensor_blueprints[0], self._camera_transforms[3], attach_to=self._parent)

        self.left_camera.listen(lambda image: CameraManager._parse_side_image(weak_self, image, 'left'))
        # self.right_camera.listen(lambda image: CameraManager._parse_side_image(weak_self, image, 'right'))
        self.rear_camera.listen(lambda image: CameraManager._parse_side_image(weak_self, image, 'rear'))

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))  # Main camera view
        if self.left_surface is not None:
            mirrored_left = pygame.transform.flip(self.left_surface, True, False)  # Flip horizontally
            display.blit(mirrored_left, (10, 800))  # Bottom-left corner
        # if self.right_surface is not None:
        #     display.blit(self.right_surface, (self.hud.dim[0] - 250, self.hud.dim[1] - 150))  # Bottom-right corner
        if self.rear_surface is not None:
            mirrored_rear = pygame.transform.flip(self.rear_surface, True, False)  # Flip horizontally
            display.blit(mirrored_rear, (1200, 325))  # Center-bottom



    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        image.convert(cc.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

    @staticmethod
    def _parse_side_image(weak_self, image, position):
        self = weak_self()
        if not self:
            return
        image.convert(cc.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if position == 'left':
            self.left_surface = pygame.transform.scale(surface, (300, 200))
        # elif position == 'right':
        #     self.right_surface = pygame.transform.scale(surface, (300, 200))
        elif position == 'rear':
            self.rear_surface = pygame.transform.scale(surface, (500, 120))




def initialize_csv(filename, headers):
    """Initialize the CSV file and write headers."""
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)  # Create a csv.writer object
        writer.writerow(headers)   # Write headers to the CSV file

def log_data(filename, steer_cmd, brake_cmd, throttle_cmd, distance):
    """Append data to the CSV file."""
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)  # Create a csv.writer object
        writer.writerow([steer_cmd, brake_cmd, throttle_cmd,distance])  # Write a row of data

##############################################################################################################################################################
##############################################################################################################################################################
class Log:
    def __init__(self):
        print("Log initialized")
        # Initialize the log with an empty DataFrame and columns
        self.cols = ['tick','time', 'steerCmd', 'brakeCmd', 'throttleCmd', 'distance1', 'distance2']
        self.df = pd.DataFrame(columns=self.cols)  # DataFrame with specific columns
        self.tick = 0  # Set the initial tick to 0
        self.clock = pygame.time.Clock()


    def log_data(self, steerCmd, brakeCmd, throttleCmd, distance1, distance2,world, increment_tick=False):
        # Create a new row with the provided data and current tick
        # timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # snapshot = world.get_snapshot()  # Correct call to get snapshot
        # simulation_time = snapshot.timestamp.elapsed_seconds  # Get elapsed time from simulation start
        # simulation_time_raw = time.time()  # Use the current time in seconds since the Epoch
        
        current_time = datetime.datetime.now()

        # Extract hours, minutes, seconds, and milliseconds
        hours = current_time.strftime('%H')
        minutes = current_time.strftime('%M')
        seconds = current_time.strftime('%S')
        milliseconds = current_time.microsecond // 1000  # Convert microseconds to milliseconds
        
        # Format the time as HH:MM:SS:SSS
        formatted_time = f"{hours}:{minutes}:{seconds}:{milliseconds:03}"
        

        new_data = {
            'tick': self.tick,
            'time': formatted_time,
            'steerCmd': steerCmd,
            'brakeCmd': brakeCmd,
            'throttleCmd': throttleCmd,
            'distance1': distance1,
            'distance2': distance2
        }
        
        # Add the new data to the DataFrame
        self.df = pd.concat([self.df, pd.DataFrame([new_data])], ignore_index=True)
        
        # Increment tick if flag is raised
        if increment_tick:
            self.tick += 1

    def destroy(self, filename='log.csv'):
        print("Log destroyed")
        # Save the log to a CSV file before destroying
        self.df.to_csv(filename, index=False)
        print(f"Log saved to {filename}")
        # self.generate_graph()
        del self  # Explicitly delete the instance to call destructor
##############################################################################################################################################################
##############################################################################################################################################################


############################################################################### to check if logs as expected
# Initialize log function (call this once before the game loop starts)
def init_log(log_filename):
    # Open CSV file and create writer object
    csvfile = open(log_filename, 'w', newline='')
    fieldnames = ['timestamp', 'steerCmd', 'brakeCmd', 'throttleCmd', 'Heading']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()  # Write CSV header
    return csvfile, writer

# Log telemetry data (call this inside the game loop on each tick)
def log_telemetry(writer, vehicle):
    # Get vehicle control and heading data
    control = vehicle.get_control()
    transform = vehicle.get_transform()
    heading = transform.rotation.yaw

    # Write current data to CSV
    writer.writerow({
        'timestamp': time.time(),
        'steerCmd': control.steer,
        'brakeCmd': control.brake,
        'throttleCmd': control.throttle,
        'Heading': heading
    })

def map_steering_input_to_degrees(x):
    # Ensure x is between -1 and 1
    if x < -1 or x > 1:
        raise ValueError("Input must be between -1 and 1")

    # Convert x from range [-1, 1] to range [-225, 225]
    return (x * 225)




# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================


def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None
    filename = "./example.csv"
    headers = ['steerCmd', 'brakeCmd', 'throttleCmd']
    initialize_csv(filename, headers)
    ### init_log(r'./example_telemetry.csv') for logging function


    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)
        # print(client.get_available_maps())

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE)

        hud = HUD(args.width, args.height)
        log = Log()

        world = World(client.get_world(), hud, args.filter, log,client)
        # world = client.load_world('Town04')
        controller = DualControl(world, args.autopilot,writer)
        clock = pygame.time.Clock()
        
        while True:
            
            clock.tick_busy_loop(60)
            v = world.player.get_velocity()
            if controller.parse_events(world, clock,v):
                return
            world.tick(clock)
            
        ###################################################################################################################### 
            # vehicle_t = world.player
            # # Calculate distance to other vehicles
            # vehicles = world.world.get_actors().filter('vehicle.*')
            # if len(vehicles) > 1:
            #     distance_func = lambda l: math.sqrt((l.x - transform.location.x)**2 + (l.y - transform.location.y)**2 + (l.z - transform.location.z)**2)
            #     nearby_vehicles = [(distance_func(v.get_location()), v) for v in vehicles if v.id != vehicle.id]
            #     nearest_distance = sorted(nearby_vehicles)[0][0] if nearby_vehicles else 0
            # else:
            #     nearest_distance = 0
            # log_telemetry(writer, vehicle_t)
        #######################################################################################################################
        
            world.render(display)
            pygame.display.flip()

    finally:

        if world is not None:
            world.destroy()

        pygame.quit()



def calculate_distance(loc1, loc2):
    dx = loc1.x - loc2.x
    dy = loc1.y - loc2.y
    dz = loc1.z - loc2.z
    return math.sqrt(dx**2 + dy**2 + dz**2)

# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1920x1080',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.nissan.micra',
        help='actor filter (default: "vehicle.nissan.*")')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]
    # print(args.res.split('x'))

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:

        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()
