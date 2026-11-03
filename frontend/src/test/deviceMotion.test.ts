import { describe, expect, it } from "vitest";
import { toDeviceMotionSample } from "../deviceMotion";

describe("device motion payload", () => {
  it("serializes seconds and acceleration in m/s² with exactly the backend fields", () => {
    const sample = toDeviceMotionSample({
      acceleration: { x: 1.2, y: -2.5, z: 9.81 },
      accelerationIncludingGravity: { x: 11.2, y: -2.5, z: 9.81 },
    }, 1250);

    expect(sample).toEqual({ t: 1.25, ax: 1.2, ay: -2.5, az: 9.81 });
    expect(Object.keys(sample || {})).toEqual(["t", "ax", "ay", "az"]);
  });

  it("uses accelerationIncludingGravity when acceleration is unavailable", () => {
    expect(toDeviceMotionSample({
      acceleration: null,
      accelerationIncludingGravity: { x: 0.1, y: 0.2, z: 9.7 },
    }, 500)).toEqual({ t: 0.5, ax: 0.1, ay: 0.2, az: 9.7 });
  });
});
