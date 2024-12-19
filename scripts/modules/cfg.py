from pydantic import BaseModel
import toml

class Config(BaseModel):
  dispersion       : float
  effective_area   : float
  baud_rate        : float
  fiber_length     : float
  n_modes          : int
  n_channels       : int
  launch_power     : float
  raman_gain       : float
  channel_spacing  : float
  center_frequency : float
  store            : bool
  pulse_shape      : int
  collision_margin : int
  n_pumps          : int 


# Deserialize TOML file into a Pydantic model
def load_toml_to_struct(filepath: str) -> Config:
    # with open(filepath, "rb") as f:
    data = toml.load(filepath)
    print(data)
    return Config(**data)

# Serialize a Pydantic model into a TOML file
def save_struct_to_toml(filepath: str, config: Config):
    with open(filepath, "w") as f:
        toml.dump(config.model_dump(), f)