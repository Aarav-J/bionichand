#include <Arduino.h>
#include <WiFi.h>
#include <WebServer.h>
#include <ArduinoJson.h>
int x;

const char* ssid     = "URHome";
const char* password = "jaincmYpib811498";
int fingerDegrees[] = {20,20,20,20};
// Create a web server on port 80
WebServer server(80);

void handleRoot() {
	String msg = "Hello from device";
	server.send(200, "text/plain", msg);
	Serial.println("[HTTP] GET / -> " + msg);
}

void handleEcho() {
	// echo query parameter ?msg=...
	String m = server.arg("msg");
  digitalWrite(16, HIGH); 
	if (m.length() == 0) m = "(empty)";
	server.send(200, "text/plain", "Echo: " + m);
	Serial.println("[HTTP] GET /echo msg=" + m);
}

void getDegrees() { 
  StaticJsonDocument<256> resp; 
  resp["index"] = fingerDegrees[0]; 
  resp["middle"] = fingerDegrees[1]; 
  resp["ring"] = fingerDegrees[2]; 
  resp["pinky"] = fingerDegrees[3]; 
  String out; 
  serializeJson(resp, out); 
  server.send(200, "application/json", out); 
}

void handleData() {
	// read raw POST body
	String body = server.arg("plain");
	// server.send(200, "text/plain", "Received " + String(body.length()) + " bytes");
	Serial.println("[HTTP] POST /data body: " + body);
  StaticJsonDocument<512> doc; 
  DeserializationError err = deserializeJson(doc, body); 
  if(err) { 
    Serial.println("Json parse failed: " + String(err.c_str())); 
    server.send(400, "application/json", "{\"error\": \"invalid json\"}"); 
    return; 
  }
  const char* finger = doc["finger"] | "none"; 
  const int angle = doc["angle"] | -1; 
  if(strcmp(finger, "index") == 0 && angle >= 0 && angle) { 
	fingerDegrees[0] = angle; 
	Serial.printf("Set index finger to %d degrees\n", angle); 
  } else if(strcmp(finger, "middle") == 0 && angle >= 0 && angle) { 
	fingerDegrees[1] = angle; 
	Serial.printf("Set middle finger to %d degrees\n", angle); 
  } else if(strcmp(finger, "ring") == 0 && angle >= 0 && angle) { 
	fingerDegrees[2] = angle; 
	Serial.printf("Set ring finger to %d degrees\n", angle); 
  } else if(strcmp(finger, "pinky") == 0 && angle >= 0 && angle) { 
	fingerDegrees[3] = angle; 
	Serial.printf("Set pinky finger to %d degrees\n", angle); 
  } else { 
	Serial.println("Invalid finger/angle");
  }

  const char* cmd = doc["cmd"] | "none"; 
  int value = doc["value"] | 0; 
  if(!doc["led"].isNull()) { 
    bool ledOn = doc["led"] | false; 
    digitalWrite(16, ledOn ? HIGH : LOW); 
    Serial.printf("cmd=%s value=%d led=%d\n", cmd, value, ledOn); 
  }
  

  
  
  StaticJsonDocument<128> resp;
  resp["status"] = "ok";
  resp["received"] = value;
  String out;
  serializeJson(resp, out);
  server.send(200, "application/json", out);

}

void setup() {
	Serial.begin(9600);
	delay(100);
  pinMode(16, OUTPUT);

	Serial.println("Starting WiFi...");
	WiFi.begin(ssid, password);

	// wait up to ~10s for connection
	int attempts = 0;
	while (WiFi.status() != WL_CONNECTED && attempts < 20) {
		delay(500);
		Serial.print('.');
		attempts++;
	}
	Serial.println();

	if (WiFi.status() == WL_CONNECTED) {
		Serial.print("Connected. IP: ");
		Serial.println(WiFi.localIP());
	} else {
		Serial.println("WiFi connection failed (continuing without network)");
	}

	// register routes
	server.on("/", HTTP_GET, handleRoot);
	server.on("/echo", HTTP_GET, handleEcho);
	server.on("/data", HTTP_POST, handleData);
	server.on("/degrees", HTTP_GET, getDegrees);

	server.onNotFound([]() {
		server.send(404, "text/plain", "Not found");
	});

	server.begin();
	Serial.println("HTTP server started");
}

void loop() {
	server.handleClient();
}

