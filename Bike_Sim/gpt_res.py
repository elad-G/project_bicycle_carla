import sys
sys.path.append("carla/dist/carla-0.9.14-py3.7-win-amd64.egg")
sys.path.append("carla")
import carla
from agents.navigation.basic_agent import BasicAgent 
from carla import Transform, Location, Rotation

client = carla.Client('localhost', 2000, worker_threads=1)
client.set_timeout(20.0)
world = client.get_world()
map = world.get_map()

settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)

try:
    bp = world.get_blueprint_library().find("vehicle.ford.crown")
    spec = world.get_spectator()
    t = Transform(Location(x=-8.514482, y=113.051308, z=0.067943), Rotation(pitch=-0.000034, yaw=89.710915, roll=0.000016))
    # spec.set_transform(t)
    hv = world.spawn_actor(bp, t)
    agent = BasicAgent(hv, 20, map_inst= map)
    destination = Location(x=-8.413572, y=203.051056, z=0.067480)
    world.debug.draw_string(destination, 'dd', life_time=10)
    agent.set_destination(destination)
    while True:
        world.tick()
        if agent.done():
            break
        hv.apply_control(agent.run_step())

finally:
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)
    [ x.destroy() for x in world.get_actors().filter("vehicle")]