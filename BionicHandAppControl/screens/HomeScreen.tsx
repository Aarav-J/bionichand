import React, { useEffect, useMemo, useState } from "react";
import { View, Text, StyleSheet, ScrollView, TouchableOpacity } from "react-native";
import { LinearGradient } from "expo-linear-gradient";
import Card from "../components/Card";
import PresetButton from "../components/PresetButton";
import FingerSlider from "../components/FingerSlider";
import { colors } from "../theme/colors";
import { fingers, fingerDisplay, Preset } from "../types";
import { useHandStore } from "../store/handStore";
// import { get } from "react-native/Libraries/TurboModule/TurboModuleRegistry";
import { getAngles, getLed, postAngles, toggleLed } from "../utils/utils";
import { FontAwesome6 } from "@expo/vector-icons";
export default function HomeScreen() {
  const currentAngles = useHandStore((s) => s.currentAngles);
  const desiredAngles = useHandStore((s) => s.desiredAngles);
  const setDesiredAngle = useHandStore((s) => s.setDesiredAngle);
  const applyPresetToDesired = useHandStore((s) => s.applyPresetToDesired);
  const setCurrentAngles = useHandStore((s) => s.setCurrentAngles);
  const ledState = useHandStore((s) => s.ledState);
  const setLedState = useHandStore((s) => s.setLedState);
  const lastUpdated = useHandStore((s) => s.lastUpdated);

  useEffect(() => { 
    const fetchInitialAngles = async () => {
      try {
        const angles_received = await getAngles();
        setCurrentAngles(angles_received as any);
        setDesiredAngle("thumb", 20);
        setDesiredAngle("index", angles_received.index);
        setDesiredAngle("middle", angles_received.middle);
        setDesiredAngle("ring", angles_received.ring);
        setDesiredAngle("pinky", angles_received.pinky);
      } catch (e) {
        console.error('Error fetching initial angles', e);
      }
    }

    const fetchLedState = async () => {
      getLed().then(state => {
        if (state !== ledState) setLedState(state);
      });
    }
    fetchInitialAngles();
  }, [])
  const presets: Preset[] = useMemo(
    () => [
      {
        id: "open",
        name: "Open",
        icon: "hand-paper",
        library: "FontAwesome6",
        angles: Object.fromEntries(fingers.map((f) => [f, 20])),
      },
      {
        id: "fist",
        name: "Fist",
        icon: "hand-fist",
        library: "FontAwesome6",
        angles: Object.fromEntries(fingers.map((f) => [f, 130])),
      },
      {
        id: "peace",
        name: "Peace",
        icon: "hand-peace",
        library: "MaterialCommunityIcons",
        angles: { index: 20, middle: 20, thumb: 130, ring: 130, pinky: 130},
      },
       {
        id: "pointUp",
        name: "Point Up",
        icon: "hand-point-up",
        library: "FontAwesome6",
        angles: { index: 130, middle: 20, thumb: 20, ring: 20, pinky: 20 },
      },
      { 
        id: "loveYou",
        name: "Love You",
        icon: "hand-love-you",
        library: "Tabler",
        angles: { thumb: 130, index: 20, middle: 130, ring: 130, pinky: 20 },
      }
    ],
    []
  );
  useEffect(() => { 
    const intervalId = setInterval(async () => { 
          try {
            const angles_received = await getAngles();
            // update only the current/device-reported angles
            setCurrentAngles(angles_received as any);
          } catch (e) {
            console.error('Error polling angles', e);
          }
    }, 30*1000);
    return () => clearInterval(intervalId);
  }, []);

  // local clock so the "seconds ago" updates every second without needing any other action
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    const tick = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(tick);
  }, []);

  const sendToDevice = async () => {
    console.log("Sending to device:", desiredAngles);
    await postAngles(desiredAngles);
    // After sending, read back current positions
    try {
      const angles_received = await getAngles();
      setCurrentAngles(angles_received as any);
    } catch (e) {
      console.error('Error fetching angles after send', e);
    }
  };

  return (
    <LinearGradient colors={[colors.bgTop, colors.bgBottom]} style={styles.container}>
      <ScrollView contentContainerStyle={styles.scroll}>
        <View style={styles.header}>
          <Text style={styles.title}>Kade Stinks</Text>
          <TouchableOpacity
            style={[styles.iconBtn, ledState ? styles.iconBtnOn : null]}
            activeOpacity={0.8}
            accessibilityRole="button"
            accessibilityLabel="Toggle light"
            onPress={async () => {
              
                // toggleLedState();
                await toggleLed();

            }}>
            <FontAwesome6 name="lightbulb" size={20} color={ledState ? '#FFD54F' : colors.buttonText} />
          </TouchableOpacity>
        </View>
        <Card>
          <View>
            <View style={styles.rowBetween}>
              <Text style={styles.cardTitle}>Current Positions</Text>
              <Text style={styles.unit}>deg</Text>
            </View>
            <View style={styles.divider} />
            <View style={styles.grid}>
              {fingers.map((f) => (
                <View key={f} style={styles.gridItem}>
                  <Text style={styles.gridLabel}>{fingerDisplay[f]}</Text>
                    <Text style={styles.gridValue}>{Math.round(currentAngles[f])}°</Text>
                </View>
              ))}
            </View>
            <View style={styles.divider} />
            <View style={styles.rowBottom}>
              <Text style={{ color: colors.textMuted, fontSize: 12 }}>Last Updated: {((new Date().getTime() - new Date(lastUpdated).getTime()) / 1000).toFixed(0)}s ago • {new Date(lastUpdated).toLocaleTimeString()}</Text>
              <TouchableOpacity onPress={async () => {
                try {
                  const angles_received = await getAngles();
                  setCurrentAngles(angles_received as any);
                } catch (e) {
                  console.error('Error fetching angles on manual refresh', e);
                }
              }}>
                <Text style={{ color: colors.textMuted, fontSize: 12, fontWeight: "600" }}>Refresh</Text>
              </TouchableOpacity>
            </View>
          </View>
        </Card>

        <Text style={styles.sectionTitle}>Preset Positions</Text>
        <ScrollView horizontal showsHorizontalScrollIndicator={false} style={{ marginBottom: 8 }}>
          <View style={{ flexDirection: "row" }}>
            {presets.map((p) => (
              <PresetButton key={p.id} label={p.name} icon={p.icon} library={p.library} onPress={() => applyPresetToDesired(p)} />
            ))}
          </View>
        </ScrollView>

        <Text style={styles.sectionTitle}>Sliders</Text>
        <Card>
          {fingers.map((f) => (
            <FingerSlider
              key={f}
              label={fingerDisplay[f]}
                value={desiredAngles[f]}
                onChange={(v) => setDesiredAngle(f, v)}
              min={0}
              max={180}
            />
          ))}
        </Card>

        <View style={{ height: 12 }} />

        <View style={styles.actions}>
          <TouchableOpacity style={styles.btn} onPress={sendToDevice}>
            <Text style={styles.btnText}>Send to Hand</Text>
          </TouchableOpacity>
        </View>

        <View style={{ height: 12 }} />
      </ScrollView>
    </LinearGradient>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, paddingTop: 40},
  scroll: { padding: 16, paddingTop: 20 },
  title: { color: colors.text, fontSize: 32, fontWeight: "800", marginBottom: 14 },
  sectionTitle: { color: colors.text, fontSize: 18, fontWeight: "700", marginVertical: 8 },
  cardTitle: { color: "#fff", fontWeight: "700", fontSize: 16 },
  unit: { color: colors.textMuted },
  divider: { height: 1, backgroundColor: "rgba(255,255,255,0.15)", marginVertical: 10 },
  rowBetween: { flexDirection: "row", justifyContent: "space-between", alignItems: "center" },
  grid: { flexDirection: "row", gap: 8, justifyContent: "space-between" },
  gridItem: {
    flex: 1,
    padding: 8,
    backgroundColor: "rgba(255,255,255,0.10)",
    borderRadius: 12,
    alignItems: "center",
  },
  gridLabel: { color: colors.textMuted, fontSize: 12 },
  gridValue: { color: colors.text, fontSize: 18, fontWeight: "700", marginTop: 4 },
  actions: { flexDirection: "row", justifyContent: "center" },
  btn: {
    backgroundColor: colors.buttonBg,
    paddingHorizontal: 18, paddingVertical: 10,
    borderRadius: 14,
    shadowColor: "#000", shadowOpacity: 0.25, shadowRadius: 6, shadowOffset: { width: 0, height: 3 },
  },
  btnText: { color: colors.buttonText, fontWeight: "700" },
  rowBottom: { flexDirection: "row", justifyContent: "space-between", alignItems: "center" },
  header: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 6,
    paddingHorizontal: 2,
  },
  iconBtn: {
    width: 40,
    height: 40,
    borderRadius: 10,
    backgroundColor: "rgba(255,255,255,0.06)",
    justifyContent: "center",
    alignItems: "center",
    shadowColor: "#000",
    shadowOpacity: 0.12,
    shadowRadius: 4,
    shadowOffset: { width: 0, height: 2 },
  },
  iconBtnOn: {
    backgroundColor: "rgba(255,213,79,0.12)",
    shadowColor: "#FFD54F",
    shadowOpacity: 0.3,
    shadowRadius: 6,
    shadowOffset: { width: 0, height: 3 },
  },
});