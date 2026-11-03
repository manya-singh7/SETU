export interface DeviceMotionSample {
  t: number;
  ax: number;
  ay: number;
  az: number;
}

type MotionAcceleration = Pick<DeviceMotionEventAcceleration, "x" | "y" | "z"> | null;
type MotionEventAccelerationData = Pick<DeviceMotionEvent, "acceleration" | "accelerationIncludingGravity">;
type CompleteAcceleration = { x: number; y: number; z: number };

function hasThreeAxes(value: MotionAcceleration): value is CompleteAcceleration {
  return value !== null
    && Number.isFinite(value.x)
    && Number.isFinite(value.y)
    && Number.isFinite(value.z);
}

/** DeviceMotionEvent acceleration values are in m/s²; elapsedMs is converted to seconds. */
export function toDeviceMotionSample(
  event: MotionEventAccelerationData,
  elapsedMs: number,
): DeviceMotionSample | null {
  const acceleration = hasThreeAxes(event.acceleration)
    ? event.acceleration
    : event.accelerationIncludingGravity;
  if (!hasThreeAxes(acceleration)) return null;
  return {
    t: elapsedMs / 1000,
    ax: acceleration.x,
    ay: acceleration.y,
    az: acceleration.z,
  };
}
