#!/usr/bin/env python3

import time
import unittest
import numpy as np
from collections import namedtuple
from smbus2 import SMBus

import cereal.messaging as messaging
from cereal import log
from system.hardware import TICI, HARDWARE
from selfdrive.test.helpers import with_processes
from selfdrive.manager.process_config import managed_processes

SENSOR_CONFIGURATIONS = (
  {
    ('bmx055', 'acceleration'),
    ('bmx055', 'gyroUncalibrated'),
    ('bmx055', 'magneticUncalibrated'),
    ('bmx055', 'temperature'),
    ('lsm6ds3', 'acceleration'),
    ('lsm6ds3', 'gyroUncalibrated'),
    ('lsm6ds3', 'temperature'),
    ('rpr0521', 'light'),
  },
  {
    ('lsm6ds3', 'acceleration'),
    ('lsm6ds3', 'gyroUncalibrated'),
    ('lsm6ds3', 'temperature'),
    ('mmc5603nj', 'magneticUncalibrated'),
    ('rpr0521', 'light'),
  },
  {
    ('bmx055', 'acceleration'),
    ('bmx055', 'gyroUncalibrated'),
    ('bmx055', 'magneticUncalibrated'),
    ('bmx055', 'temperature'),
    ('lsm6ds3trc', 'acceleration'),
    ('lsm6ds3trc', 'gyroUncalibrated'),
    ('lsm6ds3trc', 'temperature'),
    ('rpr0521', 'light'),
  },
  {
    ('lsm6ds3trc', 'acceleration'),
    ('lsm6ds3trc', 'gyroUncalibrated'),
    ('lsm6ds3trc', 'temperature'),
    ('mmc5603nj', 'magneticUncalibrated'),
    ('rpr0521', 'light'),
  },
)

Sensor = log.SensorEventData.SensorSource
SensorConfig = namedtuple('SensorConfig', ['type', 'min_samples', 'sanity_min', 'sanity_max'])
ALL_SENSORS = {
  Sensor.rpr0521: {
    SensorConfig("light", 100, 0, 150),
  },

  Sensor.lsm6ds3: {
    SensorConfig("acceleration", 100, 5, 15),
    SensorConfig("gyroUncalibrated", 100, 0, .2),
    SensorConfig("temperature", 100, 0, 60),
  },

  Sensor.lsm6ds3trc: {
    SensorConfig("acceleration", 100, 5, 15),
    SensorConfig("gyroUncalibrated", 100, 0, .2),
    SensorConfig("temperature", 100, 0, 60),
  },

  Sensor.bmx055: {
    SensorConfig("acceleration", 100, 5, 15),
    SensorConfig("gyroUncalibrated", 100, 0, .2),
    SensorConfig("magneticUncalibrated", 100, 0, 300),
    SensorConfig("temperature", 100, 0, 60),
  },

  Sensor.mmc5603nj: {
    SensorConfig("magneticUncalibrated", 100, 0, 300),
  }
}

SENSOR_BUS = 1
I2C_ADDR_LSM = 0x6A
LSM_INT_GPIO = 84


def get_proc_interrupts(int_pin):

  with open("/proc/interrupts") as f:
    lines = f.read().split("\n")

  for line in lines:
    if f" {int_pin} " in line:
      return ''.join(list(filter(lambda e: e != '', line.split(' ')))[1:6])

  return ""

def read_sensor_events(sensor_types, duration_sec):
  esocks = {}
  events = {}
  for stype in sensor_types:
    esocks[stype] = messaging.sub_sock(stype, timeout=0.1)
    events[stype] = []

  start_time_sec = time.monotonic()
  while time.monotonic() - start_time_sec < duration_sec:
    for esock in esocks:
      events[esock] += messaging.drain_sock(esocks[esock])
    time.sleep(0.01)

  for etype in events:
    assert len(events[etype]) != 0, f"No {etype} events collected"

  return events

def verify_100Hz_rate(type_name, data_list):
  data_list.sort()
  tdiffs = np.diff(data_list)

  high_delay_diffs = set(filter(lambda d: d >= 10.1*10**6, tdiffs))
  assert len(high_delay_diffs) < 10, f"Too many high delay packages: {high_delay_diffs} ({type_name})"

  avg_diff = sum(tdiffs)/len(tdiffs)
  assert avg_diff > 9.6*10**6, f"avg difference {avg_diff}, below threshold ({type_name})"

  stddev = np.std(tdiffs)
  assert stddev < 1.5*10**6, f"Standard-dev to big {stddev} ({type_name})"


class TestSensord(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    if not TICI:
      raise unittest.SkipTest

    # make sure gpiochip0 is readable
    HARDWARE.initialize_hardware()

    # read initial sensor values every test case can use
    managed_processes["sensord"].start()
    cls.events = read_sensor_events(['accelerometer', 'gyroscope', 'magnetometer',
                                     'lightSensor', 'temperatureSensor'], 5)
    managed_processes["sensord"].stop()

  def tearDown(self):
    # interrupt check might leave sensord running
    managed_processes["sensord"].stop()

  def test_sensors_present(self):
    # verify correct sensors configuration

    seen = set()
    for etype in self.events:
      for measurement in self.events[etype]:
        m = getattr(measurement, measurement.which())
        seen.add((str(m.source), m.which()))

    self.assertIn(seen, SENSOR_CONFIGURATIONS)

  def test_lsm6ds3_100Hz(self):
    # verify measurements are sampled and published at a 100Hz rate

    accel_data = set()
    gyro_data = set()
    for measurement in self.events['accelerometer'] + self.events['gyroscope']:
      m = getattr(measurement, measurement.which())

      # filter bmx events
      if not str(m.source).startswith("lsm6ds3"):
        continue

      if measurement.which() == 'accelerometer':
        accel_data.add(m.timestamp)
      else:
        gyro_data.add(m.timestamp)

    assert len(accel_data) != 0, "No lsm6ds3 accelerometer sensor events"
    assert len(gyro_data) != 0, "No lsm6ds3 gyroscope sensor events"

    verify_100Hz_rate("accelerometer", list(accel_data))
    verify_100Hz_rate("gyroscope", list(gyro_data))


  def test_events_check(self):
    # verify if all sensors produce events

    sensor_events = dict()
    for etype in self.events:
      for measurement in self.events[etype]:
        m = getattr(measurement, measurement.which())

        if m.type in sensor_events:
          sensor_events[m.type] += 1
        else:
          sensor_events[m.type] = 1

    for s in sensor_events:
      err_msg = f"Sensor {s}: 200 < {sensor_events[s]}"
      assert sensor_events[s] > 200, err_msg

  def test_logmonottime_timestamp_diff(self):
    # ensure diff between the message logMonotime and sample timestamp is small

    tdiffs = list()
    for etype in self.events:
      for measurement in self.events[etype]:
        m = getattr(measurement, measurement.which())

        # check if gyro and accel timestamps are before logMonoTime
        if str(m.source).startswith("lsm6ds3"):
          if m.which() != 'temperature':
            err_msg = f"Timestamp after logMonoTime: {m.timestamp} > {measurement.logMonoTime}"
            assert m.timestamp < measurement.logMonoTime, err_msg

        # negative values might occur, as non interrupt packages created
        # before the sensor is read
        tdiffs.append(abs(measurement.logMonoTime - m.timestamp))

    high_delay_diffs = set(filter(lambda d: d >= 10*10**6, tdiffs))
    assert len(high_delay_diffs) < 15, f"Too many high delay packages: {high_delay_diffs}"

    avg_diff = round(sum(tdiffs)/len(tdiffs), 4)
    assert avg_diff < 4*10**6, f"Avg packet diff: {avg_diff:.1f}ns"

    stddev = np.std(tdiffs)
    assert stddev < 2*10**6, f"Timing diffs have to high stddev: {stddev}"

  @with_processes(['sensord'])
  def test_sensor_values_sanity_check(self):

    events = read_sensor_events(['accelerometer', 'gyroscope', 'magnetometer',
                                 'lightSensor', 'temperatureSensor'], 2)

    sensor_values = dict()
    for etype in events:
      for m in events[etype]:
        event = getattr(m, m.which())
        key = (event.source.raw, event.which())
        values = getattr(event, event.which())

        if hasattr(values, 'v'):
          values = values.v
        values = np.atleast_1d(values)

        if key in sensor_values:
          sensor_values[key].append(values)
        else:
          sensor_values[key] = [values]

    # Sanity check sensor values and counts
    for sensor, stype in sensor_values:

      for s in ALL_SENSORS[sensor]:
        if s.type != stype:
          continue

        key = (sensor, s.type)
        val_cnt = len(sensor_values[key])
        err_msg = f"Sensor {sensor} {s.type} got {val_cnt} measurements, expected {s.min_samples}"
        assert val_cnt > s.min_samples, err_msg

        mean_norm = np.mean(np.linalg.norm(sensor_values[key], axis=1))
        err_msg = f"Sensor '{sensor} {s.type}' failed sanity checks {mean_norm} is not between {s.sanity_min} and {s.sanity_max}"
        assert s.sanity_min <= mean_norm <= s.sanity_max, err_msg

  def test_sensor_verify_no_interrupts_after_stop(self):

    managed_processes["sensord"].start()
    time.sleep(1)

    # check if the interrupts are enableds
    with SMBus(SENSOR_BUS, force=True) as bus:
      int1_ctrl_reg = bus.read_byte_data(I2C_ADDR_LSM, 0x0D)
      assert int1_ctrl_reg == 3, "Interrupts not enabled!"

    # read /proc/interrupts to verify interrupts are received
    state_one = get_proc_interrupts(LSM_INT_GPIO)
    time.sleep(1)
    state_two = get_proc_interrupts(LSM_INT_GPIO)

    assert state_one != state_two, "no Interrupts received after sensord start!"

    managed_processes["sensord"].stop()

    # check if the interrupts got disabled
    with SMBus(SENSOR_BUS, force=True) as bus:
      int1_ctrl_reg = bus.read_byte_data(I2C_ADDR_LSM, 0x0D)
      assert int1_ctrl_reg == 0, "Interrupts not disabled!"

    # read /proc/interrupts to verify no more interrupts are received
    state_one = get_proc_interrupts(LSM_INT_GPIO)
    time.sleep(1)
    state_two = get_proc_interrupts(LSM_INT_GPIO)
    assert state_one == state_two, "Interrupts received after sensord stop!"


if __name__ == "__main__":
  unittest.main()
