import React from "react";
import { TouchableOpacity, View, Text, StyleSheet } from "react-native";
import { MaterialCommunityIcons, FontAwesome6 } from "@expo/vector-icons";
import {IconHandLoveYou} from "@tabler/icons-react-native";
import { colors } from "../theme/colors";

type Props = {
  label: string;
  icon: string;
  library: string; 
  onPress: () => void;
};

export default function PresetButton({ label, icon, onPress, library}: Props) {
  return (
    <TouchableOpacity onPress={onPress} style={styles.wrap} accessibilityLabel={`Preset ${label}`}>
      <View style={styles.circle}>
        {(() => {
          switch(library) {
            case "FontAwesome6":
              return <FontAwesome6 name={icon as any} size={28} color={colors.text} />;
            case "MaterialCommunityIcons":
              return <MaterialCommunityIcons name={icon as any} size={28} color={colors.text} />;
            case "Tabler":
              return <IconHandLoveYou size={28} color={"white"}/>;
            default:
              return <FontAwesome6 name={icon as any} size={28} color={colors.text} />;
          }
        })()}
      </View>
      <Text style={styles.text}>{label}</Text>
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  wrap: { alignItems: "center", marginRight: 14 },
  circle: {
    width: 72, height: 72, borderRadius: 36,
    backgroundColor: colors.chipBg,
    alignItems: "center", justifyContent: "center",
  },
  text: { color: colors.text, marginTop: 8, fontSize: 14 },
});