import os
import trimesh
import json

# Path to FBX file
fbx_path = os.path.join("H:\\Documents\\AirSimExperiments\\Hybrid_Navigation\\Map_Reactive\\Map_Reactive.obj")  # Update as needed

# Path to settings.json
settings_path = os.path.join("C:\\Users\\Jacob\\Documents\\AirSim\\settings.json")  # Update as needed

# Load the mesh/scene
mesh = trimesh.load(fbx_path)

def get_bounds(m):
    if hasattr(m, 'geometry'):
        all_bounds = m.bounds
        obstacles = []
        for name, geom in m.geometry.items():
            obstacles.append({
                "name": name,
                "bounds": [geom.bounds[0].tolist(), geom.bounds[1].tolist()]
            })
        return all_bounds, obstacles
    else:
        return m.bounds, []

bounds, obstacles = get_bounds(mesh)

settings = {
    "map_bounds": {
        "min": bounds[0].tolist(),
        "max": bounds[1].tolist()
    },
    "obstacles": obstacles
}
with open(settings_path, "w") as f:
    json.dump(settings, f, indent=2)

print(f"Done! Written to {settings_path}")
