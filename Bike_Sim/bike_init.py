import carla
import random

class BicycleRider:
    def _init_(self, world):
        self.world = world
        self.bicycles = []
        self.bicycle_blueprint = world.get_blueprint_library().find('vehicle.bh.crossbike')

    def spawn_bicycle_rider(self, spawn_point):
        bicycle = self.world.spawn_actor(self.bicycle_blueprint, spawn_point)
        self.bicycles.append(bicycle)
        return bicycle

    def move_bicycles(self):
        for bicycle in self.bicycles:
            # Example movement: set to drive along the road at a constant speed
            control = carla.VehicleControl()
            control.throttle = 0.5
            bicycle.apply_control(control)

    def destroy_all_bicycles(self):
        for bicycle in self.bicycles:
            bicycle.destroy()
        self.bicycles = []

    def find_side_road_spawn_points(self):
        spawn_points = self.world.get_map().get_spawn_points()
        side_road_points = []
        for point in spawn_points:
            if self.is_side_road(point):
                side_road_points.append(point)
        return side_road_points

    def is_side_road(self, point):
        # Customize this function to detect whether a spawn point is on the side of the road
        # For simplicity, assume all points are on the side of the road
        return True

# Usage
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

try:
    world = client.get_world()

    # Initialize the BicycleRider class
    bicycle_rider_manager = BicycleRider()

    # Find spawn points on the side of the road
    side_road_spawn_points = bicycle_rider_manager.find_side_road_spawn_points()

    # Spawn a bicycle rider at a random side road spawn point
    if side_road_spawn_points:
        spawn_point = random.choice(side_road_spawn_points)
        bicycle_rider_manager.spawn_bicycle_rider(spawn_point)

    # Move bicycles
    while True:
        bicycle_rider_manager.move_bicycles()

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Ensure all bicycles are destroyed
    if 'bicycle_rider_manager' in locals():
        bicycle_rider_manager.destroy_all_bicycles()