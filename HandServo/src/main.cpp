#include <ESP32Servo.h>
#include <WiFi.h>
#include <WebServer.h>
#include <ArduinoJson.h>
// #include <Servo.h>

static const int SERVO_PINS[4]   = {13, 12, 14, 27};   
   
static const int MIN_US      = 500;    
static const int MAX_US      = 2500;   
static const int START_DEG   = 20;      

const char* ssid     = "URHome";
const char* password = "jaincmYpib811498";
int fingerDegrees[] = {20,20,20,20};
// Create a web server on port 80
WebServer server(80);

Servo indexFinger;
Servo middleFinger; 
Servo ringFinger;
Servo pinkyFinger;
static Servo* servos[4] = {&indexFinger, &middleFinger, &ringFinger, &pinkyFinger};

int x;

// simple, safe setter: 0..180 degrees
void setServo(int deg, Servo* servo) {
  if (deg < 0)   deg = 0;
  if (deg > 180) deg = 180;
  if (servo) servo->write(deg);
}

void handleRoot() {
	String msg = "Hello from device";
	server.send(200, "text/plain", msg);
	Serial.println("[HTTP] GET / -> " + msg);
}

void toggleLed() { 
  digitalWrite(2, !digitalRead(2)); 
  server.send(200, "text/plain", digitalRead(2) ? "LED is ON" : "LED is OFF"); 
}
void getLed() { 
	  server.send(200, "text/plain", digitalRead(2) ? "1" : "0"); 
}

void handleEcho() {
	// echo query parameter ?msg=...
	String m = server.arg("msg");
  digitalWrite(2, HIGH); 
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
	indexFinger.write(angle);
  } else if(strcmp(finger, "middle") == 0 && angle >= 0 && angle) { 
	fingerDegrees[1] = angle; 
	Serial.printf("Set middle finger to %d degrees\n", angle); 
	middleFinger.write(angle);
  } else if(strcmp(finger, "ring") == 0 && angle >= 0 && angle) { 
	fingerDegrees[2] = angle; 
	Serial.printf("Set ring finger to %d degrees\n", angle); 
	ringFinger.write(angle);
  } else if(strcmp(finger, "pinky") == 0 && angle >= 0 && angle) { 
	fingerDegrees[3] = angle; 
	Serial.printf("Set pinky finger to %d degrees\n", angle); 
	pinkyFinger.write(angle);
  } else { 
	Serial.println("Invalid finger/angle");
  }

  const char* cmd = doc["cmd"] | "none"; 
  int value = doc["value"] | 0; 
  if(!doc["led"].isNull()) { 
    bool ledOn = doc["led"] | false; 
    digitalWrite(2, ledOn ? HIGH : LOW); 
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
	Serial.begin(115200);
	delay(200);
  pinMode(2, OUTPUT);
	ESP32PWM::allocateTimer(0);
  for(int i=0; i<4; i++) { 
	servos[i]->setPeriodHertz(50); 
	int ok = servos[i]->attach(SERVO_PINS[i], MIN_US, MAX_US); 
	if(ok <= 0) { 
	  Serial.printf("ERROR: servo.attach() failed for pin %d. Check pin mapping/wiring.\n", SERVO_PINS[i]); 
	} else { 
	  Serial.printf("Servo on pin %d attached.\n", SERVO_PINS[i]); 
	} 
	setServo(START_DEG, servos[i]); 
  }
	Serial.println("Starting WiFi...");
	WiFi.begin(ssid, password);

// 	wait up to ~10s for connection
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
	server.on("/led", HTTP_GET, toggleLed);
	server.on("/getLed", HTTP_GET, getLed);
	server.onNotFound([]() {
		server.send(404, "text/plain", "Not found");
	});

	server.begin();
	Serial.println("HTTP server started");
}

void loop() {
	server.handleClient();
}

