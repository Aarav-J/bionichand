import { create } from "zustand";
import { Finger, fingers, Preset } from "../types";

type State = {
  // angles reported by the device (read-only from UI perspective)
  currentAngles: Record<Finger, number>;
  // angles the user can modify via sliders/presets, not sent to device until user presses send
  desiredAngles: Record<Finger, number>;
  lastUpdated: number; // timestamp of last update to currentAngles
  // setters for desired state (UI controls)
  setDesiredAngle: (f: Finger, value: number) => void;
  applyPresetToDesired: (p: Preset) => void;
  resetDesired: (value: number) => void;
  setLastUpdated: (timestamp: number) => void;
  // setter to update currentAngles from device polling
  ledState: boolean;
  setLedState: (state: boolean) => void;
  setCurrentAngles: (angles: Record<Finger, number>) => void;
};

const defaultAngles: Record<Finger, number> = {
  thumb: 20,
  index: 20,
  middle: 20,
  ring: 20,
  pinky: 20,
};

export const useHandStore = create<State>((set) => ({
  currentAngles: defaultAngles,
  desiredAngles: defaultAngles,
  lastUpdated: Date.now(),
  ledState: false,
  setLedState: (state: boolean) => set(() => ({ ledState: state })),
  setLastUpdated: (timestamp: number) => set(() => ({ lastUpdated: timestamp })),
  setDesiredAngle: (f, value) =>
    set((s) => ({ desiredAngles: { ...s.desiredAngles, [f]: value } })),

  applyPresetToDesired: (p) =>
    set((s) => ({
      desiredAngles: {
        ...s.desiredAngles,
        ...fingers.reduce((acc, f) => {
          const v = p.angles[f];
          if (typeof v === "number") acc[f] = v;
          return acc;
        }, {} as Record<Finger, number>),
      },
    })),

  resetDesired: (value) =>
    set(() => ({
      desiredAngles: fingers.reduce((acc, f) => ({ ...acc, [f]: value }), {} as Record<Finger, number>),
    })),

  setCurrentAngles: (angles) => set(() => ({ currentAngles: angles })),
}));