
import argparse
import pynlin
import pynlin.wdm
import pynlin.pulses
import pynlin.nlin
import pynlin.utils
import pynlin.constellations
import json
import numpy as np

def do_time_integrals(fiber_length, pulse_shape="Nyquist"):
  f = open("./scripts/sim_config.json")
  data = json.load(f)
  dispersion = data["dispersion"]
  baud_rate = data["baud_rate"]
  num_channels = data["num_channels"]
  channel_spacing = data["channel_spacing"]
  partial_collision_margin = data["partial_collision_margin"]
  wavelength = data["wavelength"]

  beta2 = pynlin.utils.dispersion_to_beta2(
      dispersion * 1e-12 / (1e-9 * 1e3), wavelength
  )

  channel_spacing = channel_spacing
  num_channels = num_channels
  baud_rate = baud_rate

  fiber = pynlin.fiber.Fiber(
      effective_area=80e-12,
      beta2=beta2
  )
  wdm = pynlin.wdm.WDM(
      spacing=channel_spacing * 1e-9,
      num_channels=num_channels,
      center_frequency=190
  )
  partial_collision_margin = 5
  points_per_collision = 10

  coi_index = [0]
  pynlin.nlin.X0mm_time_integral_WDM_selection(
      baud_rate,
      wdm,
      coi_index,
      fiber,
      fiber_length,
      "results/general_results_alt.h5",
      pulse_shape=pulse_shape,
      rolloff_factor=0.0,
      samples_per_symbol=10,
      points_per_collision=points_per_collision,
      use_multiprocessing=True,
      partial_collisions_start=partial_collision_margin,
      partial_collisions_end=partial_collision_margin,
  )
  return 0

# do_time_integrals(100)