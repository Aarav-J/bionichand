import React from "react";
import { View, Text, StyleSheet } from "react-native";
import Slider from "@react-native-community/slider";
import { colors } from "../theme/colors";

type Props = {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min?: number;
  max?: number;
};

export default function FingerSlider({ label, value, onChange, min = 0, max = 90 }: Props) {
  return (
    <View style={styles.block}>
      <View style={styles.header}>
        <Text style={styles.label}>{label}</Text>
        <Text style={styles.value}>{Math.round(value)}°</Text>
      </View>
      <Slider
        value={value}
        onValueChange={onChange}
        minimumValue={min}
        maximumValue={max}
        step={1}
        minimumTrackTintColor="#fff"
        maximumTrackTintColor="rgba(255,255,255,0.35)"
        thumbTintColor="#fff"
      />
    </View>
  );
}

const styles = StyleSheet.create({
  block: { marginBottom: 16 },
  header: { flexDirection: "row", justifyContent: "space-between", marginBottom: 6 },
  label: { color: colors.text, fontSize: 16, fontWeight: "600" },
  value: { color: colors.textMuted, fontVariant: ["tabular-nums"] },
});