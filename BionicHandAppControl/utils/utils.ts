import { useHandStore } from "../store/handStore";

const espaddress = "http://100.70.3.129/"

export const getAngles = async (): Promise<Record<string, number>> => {
    const url = espaddress + "degrees";
    // const lastUpdated = useHandStore.getState().lastUpdated;
    const setLastUpdated = useHandStore.getState().setLastUpdated;
    console.log("Fetching angles from", url);
    try {
        const res = await fetch(url);
        if (!res.ok) throw new Error(`HTTP error ${res.status}`);
        const data = await res.json(); // <-- parse JSON here
        console.log("Angles:", data);
        setLastUpdated(Date.now());
        return data;
    } catch (error) {
        console.error('Error fetching angles:', error);
        throw error;
    }
}

export const postAngles = async (angles: Record<string, number>): Promise<void> => {
    const url = espaddress + "data";

    // Send one POST per finger with body: { finger: string, angle: number }
    const entries = Object.entries(angles);
    for (const [finger, angle] of entries) {
        if (finger === "thumb") {
            continue; 
        }
        try {
            const res = await fetch(url, {
                method: "POST",
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ finger, angle }),
            });

            if (!res.ok) {
                // Log and continue (or throw if you prefer to abort on first failure)
                console.error(`Failed to POST ${finger}: HTTP ${res.status}`);
                continue;
            }

            // If the device returns JSON, parse it; otherwise ignore
            try {
                const data = await res.json();
                console.log(`Success ${finger}:`, data);
            } catch {
                console.log(`Success ${finger} (no JSON body)`);
            }
        } catch (err) {
            console.error(`Error posting ${finger}:`, err);
        }
    }
}

export const toggleLed = async () => { 
    const setLedState = useHandStore.getState().setLedState;
    const ledState = useHandStore.getState().ledState;
    const url = espaddress + "led";
    const res = await fetch(url, { method: "GET" });
    if (!res.ok) {
        throw new Error(`HTTP error ${res.status}`)
    } else { 
        setLedState(!ledState);
    }
}

export const getLed = async () => { 
    const url = espaddress + "getLed";
    // const setLedState = useHandStore.getState().setLedState;
    const res = await fetch(url, { method: "GET" });
    if (!res.ok) {
        throw new Error(`HTTP error ${res.status}`)
    }else { 
        const data = await res.json();
        return parseInt(data.ledState) !== 0;
    }
}