export type Finger = "thumb" | "index" | "middle" | "ring" | "pinky";

export const fingers: Finger[] = ["thumb", "index", "middle", "ring", "pinky"];

export const fingerDisplay: Record<Finger, string> = {
  thumb: "Thumb",
  index: "Index",
  middle: "Middle",
  ring: "Ring",
  pinky: "Pinky",
};

export type Preset = {
  library: string;
  id: string;
  name: string;
  icon: string; // vector icon name
  angles: Partial<Record<Finger, number>>; // 0–90 deg
};